# rct-env-driven-middleware
Environment-driven middleware for reinforcement learning

### Install required dependencies
torch, gym, bababababa

### run
##### set python path
export PYTHONPATH=.

##### PPO + gym
start train server
```
python gym_integrated/train_server.py  --env_config=cartpole --algo_config=ppo
```
start env client ( Can open multiple )
```
python gym_integrated/env_client.py --env_config=cartpole_ppo
```

##### A2C + gym
start train server
```
python gym_integrated/train_server.py  --env_config=cartpole --algo_config=a2c
```
start env client( Can open multiple )
```
python gym_integrated/env_client.py --env_config=cartpole
```

##### DQN + gym
start train server
```
python gym_integrated/train_server.py  --env_config=cartpole --algo_config=dqn
```
start env client( Can open multiple )
```
python gym_integrated/env_client.py --env_config=cartpole
```

##### Douzero
start train server
```
python douzero_integrated/train_server_dmc.py --env_config=doudizu --algo_config=dmc
```
start env client( Can open multiple )
```
python douzero_integrated/env_client.py --env_config=doudizu
```
