from http.server import SimpleHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import cloudpickle as pickle
import traceback


from .config.default_config import GET_ACTION, GET_INIT_HIDDEN_STATE, STORE


class PolicyServer(ThreadingMixIn, HTTPServer):
    def __init__(self, agent, address, port, handler_type='default', close_log=False):
        handler = _make_handler(handler_type=handler_type)
        handler.agent = agent
        if close_log:
            handler.log_message = lambda format, *args: None
        HTTPServer.__init__(self, (address, port), handler)


class BaseHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length"), 0)
        raw_body = self.rfile.read(content_len)
        parsed_input = pickle.loads(raw_body)
        try:
            response = self.execute_command(parsed_input)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(pickle.dumps(response))
        except Exception:
            self.send_error(500, traceback.format_exc())

    # 按照自己来继承实现
    def execute_command(self, args):
        raise NotImplementedError(f"execute_command func must implement")


class Handler(BaseHandler):
    def execute_command(self, args):
        command = args["command"]
        response = {}
        if command == GET_ACTION:
            action, hidden_state = self.agent.get_action(
                args['observation'], args['hidden_state'])
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
                args['observation'], args['hidden_state'])
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
            _action_idx = self.agent.get_action(
                args['position'], args['obs'], args['flags'])
            response = {'action': _action_idx}

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
