import argparse
import threading

from axie_origin_integrated.agent.dmc_agent import DMCV3Agent
from core.policy_server import PolicyServer
from core.base_agent import train

parser = argparse.ArgumentParser()

parser.add_argument('--ip', type=str, default='127.0.0.1', help='The ip to use.')

parser.add_argument("--port", type=int, default=6789,
                    help="The port to use (on localhost).")

parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')

parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')

parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')

parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')

parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')

parser.add_argument('--batch_size', default=100, type=int,
                    help='batch_size')

parser.add_argument('--buffer_size', default=105, type=int,
                    help='buffer_size')

parser.add_argument('--save_interval', default=600, type=int,
                    help='The time interval(second) for saving the model parameters')

parser.add_argument('--model_save_dir', default='./models/', type=str,
                    help='The model save path')

parser.add_argument('--model_load_dir', default='./models/', type=str,
                    help='The model load path')

parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')

parser.add_argument('--train_device', default='cpu', type=str) # cpu or 0, 1, 2, ...

parser.add_argument('--infer_device', default='cpu', type=str) # cpu or 0, 1, 2, ...

parser.add_argument('--close_log', default=True, type=bool)

parser.add_argument('--policy_server_num', default=3, type=int,
                    help='The number of server threads you want to launch')

parser.add_argument('--model_names', default='model:0_player-id:4|model:0_player-id:5', type=str,
                    help='The model names')
#for test
if __name__ == "__main__":
    args = parser.parse_args()

    agent = DMCV3Agent(args)

    train_thread = threading.Thread(target=train,
                                    name='learn',
                                    kwargs={'agent': agent,
                                            'args': args})

    train_thread.start()

    policy_server_list = [PolicyServer(agent=agent,
                                       address=args.ip,
                                       port=args.port + i,
                                       handler_type='dmcv3',
                                       close_log=args.close_log) for i in range(args.policy_server_num)]

    policy_server_thread_list = []
    for idx, policy_server in enumerate(policy_server_list):
        thread = threading.Thread(target=policy_server.serve_forever, name=f'policy_server_{idx}')
        thread.start()
        policy_server_thread_list.append(thread)

    policy_server_thread_list = [threading.Thread(target=policy_server.serve_forever, name=f'policy_server_{idx}') for idx, policy_server in enumerate(policy_server_list)]


