import torch

from abc import ABC, abstractmethod
from typing import Dict, Union, List, Tuple

# --- HTTP ---
from tritonclient.http import InferenceServerClient as HttpInferenceServerClient
from tritonclient.http import InferInput, InferRequestedOutput

# --- gRPC ---
from tritonclient.grpc import InferenceServerClient as GrpcInferenceServerClient
from tritonclient.grpc import service_pb2


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger("TritonInferenceClient")


class BaseTritonClient(ABC):
    """
    Base class for Triton Inference Client using gRPC or HTTP.
    """

    def __init__(self,
        server_url: str,
        model_name: str,
        model_version: str = "1",
        device: str = "cuda",
        verbose: bool = False
    ):
        self.model_name = model_name
        self.model_version = model_version
        self._client = self._initialize_client(server_url, verbose)
        self._device = device

        if not self._client.is_server_live():
            raise ConnectionError("Triton server is not live. Ensure the server is running.")
        LOGGER.info(f"Triton server at {server_url} is live!")

        if not self._client.is_model_ready(self.model_name, self.model_version):
            raise RuntimeError(f"Model {self.model_name} (v{self.model_version}) is not ready.")
        LOGGER.info(f"Model {self.model_name} (v{self.model_version}) is ready!")

        model_metadata = self._client.get_model_metadata(self.model_name, self.model_version)
        self._inputs_metadata = model_metadata.get('inputs', [])
        self._outputs_metadata = model_metadata.get('outputs', [])
        self._input_names: List[str] = [meta['name'] for meta in self._inputs_metadata]

        LOGGER.info(f"Initialize {self.__class__.__name__} for model '{self.model_name}'")

    @abstractmethod
    def _initialize_client(self, server_url: str, verbose: bool):
        """
        Initialize the Triton Inference Client using the specific protocol.
        """
        pass

    @abstractmethod
    def infer(self, inputs_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform inference using the specific protocol.
        """
        pass

    def __call__(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Process arguments and call the protocol-specific infer method.
        """
        if len(args) > len(self._input_names):
            raise ValueError(
                f"Receive {len(args)} positional arguments, but model '{self.model_name}' "
                f"only accepts {len(self._input_names)}. Valid inputs are: {self._input_names}"
            )

        inputs_data: Dict[str, torch.Tensor] = {}

        # Positional Arguments
        for i, arg in enumerate(args):
            input_name = self._input_names[i]
            inputs_data[input_name] = arg

        # Keyword Arguments
        for name, data in kwargs.items():
            if name not in self._input_names:
                raise ValueError(
                    f"'{name}' is not a valid input name for model '{self.model_name}'. "
                    f"Valid names are: {self._input_names}"
                )
            if name in inputs_data:
                raise ValueError(
                    f"Receive duplicate value for input '{name}'. "
                    "It was provided as both a positional and keyword argument."
                )
            inputs_data[name] = data

        # Check if all required inputs are present
        required_inputs = set(self._input_names)
        provided_inputs = set(inputs_data.keys())
        if required_inputs != provided_inputs:
            missing = sorted(list(required_inputs - provided_inputs))
            raise ValueError(f"Missing required inputs for model '{self.model_name}': {missing}")

        outputs_dict = self.infer(inputs_data)
        outputs = [outputs_dict[meta['name']] for meta in self._outputs_metadata]

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def close(self):
        """
        Close the connection to the server.
        """
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class TritonGrpcClient(BaseTritonClient):
    """
    Triton Inference Client using the gRPC protocol.
    """

    def _initialize_client(self, server_url: str, verbose: bool):
        return GrpcInferenceServerClient(url=server_url, verbose=verbose)

    def infer(self, inputs_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        request = service_pb2.ModelInferRequest()
        request.model_name = self.model_name
        request.model_version = self.model_version

        raw_input_contents = []
        for meta in self._inputs_metadata:
            name = meta['name']
            data = inputs_data[name]

            input_tensor_meta = service_pb2.ModelInferRequest().InferInputTensor()
            input_tensor_meta.name = name
            input_tensor_meta.datatype = meta['datatype']
            input_tensor_meta.shape.extend(list(data.shape))
            request.inputs.extend([input_tensor_meta])
            
            numpy_data = data.cpu().numpy()
            raw_input_contents.append(numpy_data.tobytes())

        request.raw_input_contents.extend(raw_input_contents)

        for meta in self._outputs_metadata:
            output_tensor_meta = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            output_tensor_meta.name = meta['name']
            request.outputs.extend([output_tensor_meta])

        response = self._client.infer(
            model_name=self.model_name,
            request_body=request,
            model_version=self.model_version
        )
        
        outputs = {}
        for meta in self._outputs_metadata:
            name = meta['name']
            numpy_output = response.as_numpy(name)
            outputs[name] = torch.from_numpy(numpy_output).to(self._device)
        return outputs


class TritonHttpClient(BaseTritonClient):
    """
    Triton Inference Client using the HTTP protocol.
    """

    def _initialize_client(self, server_url: str, verbose: bool):
        return HttpInferenceServerClient(url=server_url, verbose=verbose)

    def infer(self, inputs_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        infer_inputs = []
        for meta in self._inputs_metadata:
            name = meta['name']
            data = inputs_data[name]
            numpy_data = data.cpu().numpy()
            
            infer_input = InferInput(name, list(numpy_data.shape), meta['datatype'])
            infer_input.set_data_from_numpy(numpy_data, binary_data=True)
            infer_inputs.append(infer_input)

        requested_outputs = [
            InferRequestedOutput(meta['name'], binary_data=True)
            for meta in self._outputs_metadata
        ]

        response = self._client.infer(
            model_name=self.model_name,
            inputs=infer_inputs,
            outputs=requested_outputs,
            model_version=self.model_version
        )

        outputs = {}
        for meta in self._outputs_metadata:
            name = meta['name']
            numpy_output = response.as_numpy(name)
            outputs[name] = torch.from_numpy(numpy_output).to(self._device)
        return outputs