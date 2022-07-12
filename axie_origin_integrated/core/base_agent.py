import threading
import time


class BaseAgent:
    def store(self, data):
        raise NotImplementedError(f"store func must implement")

    def update_infer_model(self):
        raise NotImplementedError(f"update_infer_model func must implement")

    def get_action(self, obs, hidden_state, infer_model_id=0):
        raise NotImplementedError(f"get_action func must implement")

    def get_init_hidden_state(self):
        raise NotImplementedError(f"get_init_hidden_state func must implement")

    def check_update(self):
        raise NotImplementedError(f"check_update func must implement")

    def train(self):
        raise NotImplementedError(f"train func must implement")

    def model_save(self):
        raise NotImplementedError(f"model_save func must implement")


def train(agent, args=None, lock=threading.Lock()):
    last_checkpoint_time = time.time()

    while True:
        time.sleep(1)
        agent.store()
        if agent.check_update():
            agent.train()
            with lock:
                agent.update_infer_model()

            if args and (time.time() - last_checkpoint_time > args.save_interval):
                last_checkpoint_time = time.time()
                agent.model_save()

def axiev3_train(agent, args=None, lock=threading.Lock()):
    last_checkpoint_time = time.time()

    while True:
        time.sleep(1)
        agent.store()

        ready_list = agent.check_update()
        agent.train(ready_list=ready_list)
        with lock:
            agent.update_infer_model(ready_list=ready_list)

        if args and (time.time() - last_checkpoint_time > args.save_interval):
            last_checkpoint_time = time.time()
            agent.model_save()
