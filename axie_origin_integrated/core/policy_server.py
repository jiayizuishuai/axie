import grpc
from concurrent import futures
from rl_grpc.rl_grpc_string import service_pb2_grpc, service_pb2
from config.default_config import GET_ACTION, GET_INIT_HIDDEN_STATE, STORE
import json

class PolicyServer:
    def __init__(self, agent, address, port,
                 handler_type='default',
                 close_log=False,
                 infer_model_id=0,
                 max_workers=None):
        handler = _make_handler(handler_type=handler_type)
        handler.agent = agent
        handler.infer_model_id = infer_model_id
        if close_log:
            handler.log_message = lambda format, *args: None

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        service_pb2_grpc.add_ModelServiceServicer_to_server(handler(), self.server)
        self.server.add_insecure_port("{}:{}".format(address, port))

    def serve_forever(self):
        print("Server started. Awaiting jobs...")
        self.server.start()
        self.server.wait_for_termination()


class BaseHandler(service_pb2_grpc.ModelServiceServicer):
    def Call(self, request, context):
        parsed_input = request.input
        response = self.execute_command(json.loads(parsed_input))
        if not response:
            response = parsed_input
        output_data = json.dumps(response)
        return service_pb2.CallResponse(output_specs=output_data)


    # 按照自己来继承实现
    def execute_command(self, args):
        raise NotImplementedError(f"execute_command func must implement")


class Handler(BaseHandler):
    def execute_command(self, args):
        command = args["command"]
        response = {}
        if command == GET_ACTION:
            action, hidden_state = self.agent.get_action(
                args['observation'], args['hidden_state'], infer_model_id=self.infer_model_id)
            response = {'action': action, 'hidden_state': hidden_state}
        elif command == GET_INIT_HIDDEN_STATE:
            response = {'hidden_state': None}
        elif command == STORE:
            self.agent.store(args['episode_data'])
        else:
            raise Exception("Unknown command: {}".format(command))
        return response


class PPOHandler(BaseHandler):
    def execute_command(self, args):
        command = args["command"]
        response = {}
        if command == GET_ACTION:
            action, hidden_state, action_prob = self.agent.get_action(
                args['observation'], args['hidden_state'], infer_model_id=self.infer_model_id)
            response = {'action': action,
                        'hidden_state': hidden_state, 'action_prob': action_prob}
        elif command == GET_INIT_HIDDEN_STATE:
            response = {'hidden_state': None}
        elif command == STORE:
            self.agent.store(args['episode_data'])
        else:
            raise Exception("Unknown command: {}".format(command))
        return response


class DMCHandler(BaseHandler):
    def execute_command(self, args):
        command = args["command"]
        response = {}
        if command == GET_ACTION:
            _action_idx = self.agent.get_action(
                args["position"], args["z_batch"], args["x_batch"], args["flags"])
            response = {'action': _action_idx}
        elif command == GET_INIT_HIDDEN_STATE:
            response = {'hidden_state': None}
        elif command == STORE:
            self.agent.store(args['episode_data'], args['role'])
        else:
            raise Exception("Unknown command: {}".format(command))
        return response

class DMCV3Handler(BaseHandler):
    def execute_command(self, args):
        command = args['command']
        response = {}
        if (command == GET_ACTION):
            response = self.agent.get_action(position=args['position'],
                                             data=args['data'],
                                             flags=args['flags'])
            response = {'response': response}

        elif (command == GET_INIT_HIDDEN_STATE):
            response = {'hidden_state': None}
        elif (command == STORE):
            self.agent.store(args['episode_data'], args['role'])
        else:
            raise Exception("Unknown command: {}".format(command))

        return response


def _make_handler(handler_type='default'):
    assert handler_type in ['default', 'ppo', 'dmc', 'dmcv3'], "unsupported handler_type"

    if handler_type == 'default':
        return Handler
    elif handler_type == 'ppo':
        return PPOHandler
    elif handler_type == 'dmc':
        return DMCHandler
    elif handler_type == 'dmcv3':
        return DMCV3Handler
    else:
        raise Exception(f"Unknown handler: {handler_type}")
