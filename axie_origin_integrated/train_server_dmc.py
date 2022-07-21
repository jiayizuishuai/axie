import argparse
import threading
from multiprocessing import Process, Queue
#加载了两个对象，一个learner，一个接口模型
from agent.dmc_agent import DMCV3Agent   #这个是learner，包含更新，训练
from agent.dmc_infer_agent import DMCV3InferAgent


from core.policy_server import PolicyServer
from core.base_agent import axiev3_train
import os
import pandas as pd


import time

parser = argparse.ArgumentParser()

parser.add_argument('--ip', type=str, default='127.0.0.1', help='The ip to use.')

parser.add_argument("--port", type=int, default=5789,
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

parser.add_argument('--batch_size', default=32, type=int,
                    help='batch_size')

parser.add_argument('--buffer_size', default=1050, type=int,
                    help='buffer_size')

parser.add_argument('--save_interval', default=600, type=int,
                    help='The time interval(second) for saving the model parameters')#10分钟保存一次

parser.add_argument('--log_interval', default=10, type=int,
                    help='The time interval(second) for log the train agent stats')

parser.add_argument('--model_save_dir', default='./models/', type=str,
                    help='The model save path')

parser.add_argument('--model_load_dir', default='./models/', type=str,
                    help='The model load path')

parser.add_argument('--load_model_server', default='True',
                    help='Load an existing model')

parser.add_argument('--load_model_env', default='True',
                    help='Load an existing model')

parser.add_argument('--train_device', default='0', type=str) # cpu or 0, 1, 2, ...

parser.add_argument('--infer_device', default='cpu', type=str) # cpu or 0, 1, 2, ...

parser.add_argument('--close_log', default=False, type=bool)

parser.add_argument('--policy_server_num', default=2, type=int,
                    help='The number of server threads you want to launch')

parser.add_argument('--agent_type', default='train', type=str,
                    help='train: this agent is used as training agent, can infering and training; '
                         'service: this agent has no training service')




def create_infer(args, data_queue_dict, model_queue_dict, save_model_queue,port):
    #在这里只注册了一个主模型
    agent_infer = DMCV3InferAgent(args=args,
                            data_queue_dict=data_queue_dict,
                            model_queue_dict=model_queue_dict,
                            save_model_queue=save_model_queue)
    policy_server = PolicyServer(agent=agent_infer,#服务端加载了一个这样的对象
                                 address=args.ip,#对应的ip
                                 port=port,#对应的端口
                                 handler_type='dmcv3',
                                 close_log=args.close_log)
    policy_server.serve_forever()


# gRPC version
if __name__ == "__main__":
    args = parser.parse_args()

    num_parallel = args.policy_server_num

    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'env_src/envs_axie/config')

    model_names = pd.read_csv(os.path.join(config_path, 'positions_info.csv'))['positions'].values.tolist()

    args.model_names = model_names

    infer_agent_list = []
    data_queue_dict_list = [{model_name: Queue() for model_name in model_names} for _ in range(num_parallel)]
    model_queue_dict_list = [{model_name: Queue() for model_name in model_names} for _ in range(num_parallel)]
    save_model_queue = Queue()


    for i, (data_queue_dict, model_queue_dict) in enumerate(zip(data_queue_dict_list, model_queue_dict_list)):
        process = Process(target=create_infer, kwargs={'args': args,
                                                       'data_queue_dict': data_queue_dict,
                                                       'model_queue_dict': model_queue_dict,
                                                       'save_model_queue':save_model_queue,
                                                       'port': args.port + i})
        process.start()
        time.sleep(0.5)

    agent = DMCV3Agent(args, data_queue_dict_list, model_queue_dict_list,save_model_queue)

    '''
    #创建一个sever用来与客户端通信来得到是否保存模型
    policy_server_save_model = PolicyServer(agent=agent,
                                 address=args.ip,
                                 port=7788,
                                 handler_type='dmcv3',
                                 close_log=args.close_log)
    policy_server_save_model.serve_forever()
    '''
    #在服务端只注册主模型

    # 更新参数，infer解除阻塞
    #agent.update_infer_model()

    axiev3_train(agent,args)
    '''
    train_thread = threading.Thread(target=axiev3_train,
                                    name='learn',
                                    kwargs={'agent': agent,
                                            'args': args})
    train_thread.start()
    '''


