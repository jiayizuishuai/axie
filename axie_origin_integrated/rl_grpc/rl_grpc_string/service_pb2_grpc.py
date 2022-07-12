# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from rl_grpc.rl_grpc_string import service_pb2 as rl__grpc_dot_service__pb2


class ModelServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Call = channel.unary_unary(
                '/rl_grpc.ModelService/Call',
                request_serializer=rl__grpc_dot_service__pb2.CallRequest.SerializeToString,
                response_deserializer=rl__grpc_dot_service__pb2.CallResponse.FromString,
                )


class ModelServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Call(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Call': grpc.unary_unary_rpc_method_handler(
                    servicer.Call,
                    request_deserializer=rl__grpc_dot_service__pb2.CallRequest.FromString,
                    response_serializer=rl__grpc_dot_service__pb2.CallResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rl_grpc.ModelService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ModelService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Call(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/rl_grpc.ModelService/Call',
            rl__grpc_dot_service__pb2.CallRequest.SerializeToString,
            rl__grpc_dot_service__pb2.CallResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
