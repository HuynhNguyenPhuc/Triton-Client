import gc
from typing import Dict, Any, Union, List, Tuple

import torch

from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import shared_memory as shm

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

LOGGER = logging.getLogger("Triton Inference Client")

TRITON_TO_TORCH_DTYPE_MAP = {
    "BOOL": torch.bool,
    "UINT8": torch.uint8, "INT8": torch.int8, "INT16": torch.int16, "INT32": torch.int32, "INT64": torch.int64,
    "FP16": torch.float16, "FP32": torch.float32, "FP64": torch.float64
}

def triton2torch(triton_dtype: str) -> torch.dtype:
    """
    Map from Triton data type to Pytorch dtype
    """
    if triton_dtype not in TRITON_TO_TORCH_DTYPE_MAP:
        raise NotImplementedError(f"Triton data type '{triton_dtype}' is not supported.")
    return TRITON_TO_TORCH_DTYPE_MAP[triton_dtype]

class TritonInferenceClient:
    """
    Triton Client with shared memory
    """
    def __init__(self,
        server_url: str,
        model_name: str,
        model_version: str = "1",
        max_batch_size: int = 32,
        verbose: bool = False
    ):
        self.model_name = model_name
        self.model_version = model_version
        self._client = InferenceServerClient(url=server_url, verbose=verbose)
        self._max_batch_size = max_batch_size

        if not self._client.is_server_live():
            raise Exception("Triton server is not live. Ensure the server is running.")
        LOGGER.info("Triton server is live!")

        if not self._client.is_model_ready(self.model_name, self.model_version):
            raise Exception(f"Model {self.model_name} (version {self.model_version}) is not ready on the server.")
        LOGGER.info(f"Model {self.model_name} (version {self.model_version}) is ready!")

        model_metadata = self._client.get_model_metadata(self.model_name, self.model_version)
        self._inputs_metadata = model_metadata.get('inputs', [])
        self._outputs_metadata = model_metadata.get('outputs', [])
        self._input_names: List[str] = [meta['name'] for meta in self._inputs_metadata]

        self._input_buffer_cache: Dict[str, torch.Tensor] = {}
        self._output_buffer_cache: Dict[str, torch.Tensor] = {}
        self._shm_handle_cache: Dict[str, Any] = {}

        self._initialize_cuda_buffers()
        LOGGER.info("Triton Inference Client is ready!")

    def _initialize_cuda_buffers(self):
        """
        Pre-allocate CUDA buffers for inputs and outputs
        """
        try:
            for meta in self._inputs_metadata:
                if any(int(d) == -1 for d in meta['shape']):
                    LOGGER.warning(f"Input '{meta['name']}' has dynamic shape. Skipping pre-allocation.")
                    continue

                shape = [self._max_batch_size] + [int(d) for d in meta['shape'][1:]]
                self._get_shm_region(f"input_{meta['name']}", shape, meta['datatype'], self._input_buffer_cache)

            for meta in self._outputs_metadata:
                if any(int(d) == -1 for d in meta['shape']):
                    LOGGER.warning(f"Output '{meta['name']}' has dynamic shape. Skipping pre-allocation.")
                    continue

                shape = [self._max_batch_size] + [int(d) for d in meta['shape'][1:]]
                self._get_shm_region(f"output_{meta['name']}", shape, meta['datatype'], self._output_buffer_cache)
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to initialize CUDA buffers: {e}") from e

    def _get_shm_region(self, 
        region_base_name: str, 
        shape: List[int], 
        triton_dtype: str, 
        buffer_cache: Dict
    ):
        """
        Helper function to create and register a shared memory region.
        """
        dtype = triton2torch(triton_dtype)
        buffer = torch.empty(shape, dtype=dtype, device='cuda')
        handle_name = f"{region_base_name}_{id(self)}"
        
        handle, byte_size = shm.create_shared_memory_region_from_tensor(buffer)
        self._client.register_cuda_shared_memory(handle_name, handle, 0, byte_size)
        
        buffer_cache[region_base_name.split('_')[1]] = buffer
        self._shm_handle_cache[handle_name] = handle

    def __call__(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Perform inference on the model.
        """
        inputs_data: Dict[str, torch.Tensor] = {} 

        if len(args) > len(self._input_names): 
             raise TypeError(f"__call__() takes at most {len(self._input_names)} positional arguments but {len(args)} were given") 

        for i, arg in enumerate(args): 
            input_name = self._input_names[i] 
            inputs_data[input_name] = arg 

        for name, data in kwargs.items(): 
            if name not in self._input_names: 
                raise TypeError(f"__call__() got an unexpected keyword argument '{name}'") 
            if name in inputs_data: 
                raise TypeError(f"__call__() got multiple values for argument '{name}'") 
            inputs_data[name] = data

        if len(inputs_data) != len(self._input_names):
            missing_inputs = set(self._input_names) - set(inputs_data.keys())
            raise TypeError(f"__call__() missing {len(missing_inputs)} required positional argument(s): {', '.join(missing_inputs)}")

        grpc_inputs = []
        grpc_outputs = []
        current_batch_size = next(iter(inputs_data.values())).shape[0]

        if current_batch_size > self._max_batch_size:
            LOGGER.warning(f"Current batch size {current_batch_size} exceeds pre-configured max_batch_size {self._max_batch_size}.")

        for meta in self._inputs_metadata:
            name = meta['name']
            data = inputs_data[name]
            handle_name = f"input_{name}_{id(self)}"
            
            cached_buffer = self._input_buffer_cache.get(name)
            if cached_buffer is None or cached_buffer.numel() < data.numel():
                LOGGER.warning(f"Input '{name}' data size ({data.numel()}) > cached size ({cached_buffer.numel() if cached_buffer is not None else 0}). Re-allocating.")
                if handle_name in self._shm_handle_cache:
                    self._client.unregister_shared_memory(handle_name)
                self._get_shm_region(f"input_{name}", list(data.shape), meta['datatype'], self._input_buffer_cache)

            buffer = self._input_buffer_cache[name]
            buffer.view(-1)[:data.numel()] = data.flatten()
            
            grpc_input = InferInput(name, list(data.shape), meta['datatype'])
            grpc_input.set_shared_memory(handle_name, data.nbytes)
            grpc_inputs.append(grpc_input)

        for meta in self._outputs_metadata:
            name = meta['name']
            handle_name = f"output_{name}_{id(self)}"
            
            required_shape = [current_batch_size] + [int(d) if d != -1 else 1 for d in meta['shape'][1:]]
            required_numel = torch.Size(required_shape).numel()

            cached_buffer = self._output_buffer_cache.get(name)
            if cached_buffer is None or cached_buffer.numel() < required_numel:
                LOGGER.warning(f"Required output '{name}' size ({required_numel}) > cached size ({cached_buffer.numel() if cached_buffer is not None else 0}). Re-allocating.")
                if handle_name in self._shm_handle_cache:
                    self._client.unregister_shared_memory(handle_name)
                
                new_shape = [max(current_batch_size, self._max_batch_size)] + required_shape[1:]
                self._get_shm_region(f"output_{name}", new_shape, meta['datatype'], self._output_buffer_cache)

            buffer = self._output_buffer_cache[name]
            grpc_output = InferRequestedOutput(name)
            grpc_output.set_shared_memory(handle_name, buffer.nbytes)
            grpc_outputs.append(grpc_output)

        self._client.infer(model_name=self.model_name, model_version=self.model_version, inputs=grpc_inputs, outputs=grpc_outputs)

        outputs = tuple(
            self._output_buffer_cache[meta['name']][:current_batch_size]
            for meta in self._outputs_metadata
        )

        return outputs[0] if len(outputs) == 1 else outputs

    def close(self):
        """
        Clean up all resources.
        """
        for handle_name in self._shm_handle_cache.keys():
            try:
                self._client.unregister_shared_memory(handle_name)
            except Exception as e:
                LOGGER.error(f"Failed to unregister shared memory '{handle_name}': {e}")
        self._shm_handle_cache.clear()
        gc.collect()

    def __enter__(self):
        """
        Enter method for 'with' statement.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit method for 'with' statement.
        """
        self.close()

    def __del__(self):
        """
        Destructor method.
        """
        self.close()