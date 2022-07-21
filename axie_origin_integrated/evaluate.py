




def evaluate(env,evaluate_model_name):
    evaluate = True
    if evaluate  == True:
        win = 0
        for i in range(128):
            env.reset()
            while not env.sim.is_finished():
                env.sim.begin_round()

                current_player_index = None
                if (env.sim.battle.current_player is env.players[0]):
                    current_player_index = 0
                else:
                    current_player_index = 1
                while (not env.sim.is_finished()):
                    data = env.get_state(list_type=True),
                    action_id = evaluate_model_name.get_action(data=data,flags={'data_type': 'code'})





                    action_card, action_target = data[0]['legal_actions'][action_id]

                    # break the loop if the action is "end turn"
                    if (action_card == 'end_turn'): break

                    # execute this step in env
                    _, _, done, _ = env.step(action_card=action_card,
                                             action_target=action_target,
                                             player_index=current_player_index)

                    if (done): break

                    # end this turn
                env.sim.end_round()


            info = {'episode_return': env.get_reward(0)}
            if info['episode_return']  == 1:
                win += 1
        win = win/128
        print('胜率为：'+str(win))



