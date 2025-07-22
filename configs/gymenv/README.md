# Separate folder for FlyCraft config files

`env_config_for_ppo_easy.json` was copied from [here](https://github.com/GongXudong/fly-craft-examples/blob/main/configs/env/VVCGym/env_config_for_ppo_easy.json) as an example.

Details for FlyCraft config files are [here](https://github.com/GongXudong/fly-craft/tree/main?tab=readme-ov-file) under the "Configuration Details" of the readme.

(Below is copied from their github for easy referencing)

## Configuration Details

[Here](https://github.com/GongXudong/fly-craft/tree/main/flycraft/configs/MR_for_HER.json) is an example of the configuration, which consists of 4 blocks:

### Task

The configurations about task and simulator, including:

* **control_mode** Str: the model to be trained, *guidance_law_mode* for guidance law model, *end_to_end_mode* for end-to-end model
* **step_frequence** Int (Hz): simulation frequency.
* **max_simulate_time** Int (s): maximum simulation time, max_simulate_time * step_frequence equals maximum length of an episode.
* **h0** Int (m): initial altitude of the aircraft.
* **v0** Int (m/s): initial true air speed of the aircraft.

### Desired Goal

The configurations about the definition and sampling method of the desired goal, including:

* **use_fixed_goal** Boolean: whether to use a fixed desired goal.
* **goal_v** Float (m/s): the true air speed of the fixed desired goal.
* **goal_mu** Float (deg): the flight path elevator angle of the fixed desired goal.
* **goal_chi** Float (deg): the flight path azimuth angle of the fixed desired goal.
* **sample_random** Boolean: if don't use fixed desired goal, whether sample desired goal randomly from ([v_min, v_max], [mu_min, mu_max], [chi_min, chi_max])
* **v_min** Float (m/s): the min value of true air speed of desired goal.
* **v_max** Float (m/s): the max value of true air speed of desired goal.
* **mu_min** Float (deg): the min value of flight path elevator angle of desired goal.
* **mu_max** Float (deg): the max value of flight path elevator angle of desired goal.
* **chi_min** Float (deg): the min value of flight path azimuth angle of desired goal.
* **chi_max** Float (deg): the max value of flight path azimuth angle of desired goal.
* **available_goals_file** Str: path of the file of available desired goals. If don't use fixed desired goal and don't sample desired goal randomly, then sample desired goal from the file of available desired goals. The file is a .csv file that has at least four columns: v, mu, chi, length. The column 'length' is used to indicate whether the desired goal represented by the row can be achieved by an expert. If it can be completed, it represents the number of steps required to achieved the desired goal. If it cannot be completed, the value is 0.
* **sample_reachable_goal** Boolean: when sampling desired goals from *available_goals_file*, should only those desired goals with length>0 be sampled.
* **sample_goal_noise_std** Tuple[Float]: a tuple with three float. The standard deviation used to add Gaussian noise to the true air speed, flight path elevation angle, and flight path azimuth angle of the sampled desired goal.

### Rewards

The configurations about rewards, including:

* **dense** Dict: The configurations of the dense reward that calculated by the error on angle and on the true air speed
  * *use* Boolean: whether use this reward;
  * *b* Float: indicates the exponent used for each reward component;
  * *angle_weight* Float [0.0, 1.0]: the coefficient of the angle error component of reward;
  * *angle_scale* Float (deg): the scalar used to scale the error in direction of velocity vector;
  * *velocity_scale* Float (m/s): the scalar used to scale the error in true air speed of velocity vector.
* **dense_angle_only** Dict: The configurations of the dense reward that calculated by the error on angle only
  * *use* Boolean: whether use this reward;
  * *b* Float: indicates the exponent used for each reward component;
  * *angle_scale* Float (deg): the scalar used to scale the error in direction of velocity vector.
* **sparse** Dict: The configurations of the sparse reward
  * *use* Boolean: whether use this reward;
  * *reward_constant* Float: the reward when achieving the desired goal.

### Terminations

The configurations about termination conditions, including:

* **RT** Dict: The configurations of the Reach Target Termination (used by non-Markovian reward)
  * *use* Boolean: whether use this termination;
  * *integral_time_length* Integer (s): the number of consecutive seconds required to achieve the accuracy of determining achievement;
  * *v_threshold* Float (m/s): the error band used to determine whether true air speed meets the requirements;
  * *angle_threshold* Float (deg): the error band used to determine whether the direction of velocity vector meets the requirements;
  * *termination_reward* Float: the reward the agent receives when triggering RT.
* **RT_SINGLE_STEP** Dict: The configurations of the Reach Target Termination (used by Markovian reward, judge achievement by the error of true airspeed and the error of angle of velocity)
  * *use* Boolean: whether use this termination;
  * *v_threshold* Float (m/s): the error band used to determine whether true air speed meets the requirements;
  * *angle_threshold* Float (deg): the error band used to determine whether the direction of velocity vector meets the requirements;
  * *termination_reward* Float: the reward the agent receives when triggering RT_SINGLE_STEP.
* **RT_V_MU_CHI_SINGLE_STEP** Dict: The configurations of the Reach Target Termination (used by Markovian reward, judge achievement by the error of true airspeed, the error of flight path elevator angle, and the error of flight path azimuth angle)
  * *use* Boolean: whether use this termination;
  * *v_threshold* Float (m/s): the error band used to determine whether true air speed meets the requirements;
  * *angle_threshold* Float (deg): the error band used to determine whether the direction of velocity vector meets the requirements;
  * *termination_reward* Float: the reward the agent receives when triggering RT_V_MU_CHI_SINGLE_STEP.
* **C** Dict: The configurations of Crash Termination
  * *use* Boolean: whether use this termination;
  * *h0* Float (m): the altitude threshold below which this termination triggers;
  * *is_termination_reward_based_on_steps_left* Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * *termination_reward* Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
* **ES** Dict: The configurations of Extreme State Termination
  * *use* Boolean: whether use this termination;
  * *v_max* Float (m/s): the maximum value of true air speed. when the true air speed exceeding this value, this termination triggers;
  * *p_max* Float (deg/s): the maximum value of roll angular speed. when the roll angular speed exceeding this value, this termination triggers;
  * *is_termination_reward_based_on_steps_left* Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * *termination_reward* Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
* **T** Dict: The configurations of Timeout Termination
  * *use* Boolean: whether use this termination;
  * *termination_reward* Float: the reward when triggers this termination.
* **CMA** Dict: The configurations of Continuously Move Away Termination
  * *use* Boolean: whether use this termination;
  * *time_window* Integer (s): the time window used to detect whether this termination condition will be triggered;
  * *ignore_mu_error* Float (deg): when the error of flight path elevator angle is less than this value, the termination condition will no longer be considered;
  * *ignore_chi_error* Float (deg): when the error of flight path azimuth angle is less than this value, the termination condition will no longer be considered;
  * *is_termination_reward_based_on_steps_left* Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * *termination_reward* Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
* **CR** Dict: The configurations of Continuously Roll Termination
  * *use* Boolean: whether use this termination;
  * *continuousely_roll_threshold* Float (deg): when the angle of continuous roll exceeds this value, this termination condition is triggered;
  * *is_termination_reward_based_on_steps_left* Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * *termination_reward* Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
* **NOBR** Dict: The configurations of Negative Overload and Big Roll Termination
  * *use* Boolean: whether use this termination;
  * *time_window* Integer (s): the time window used to detect whether this termination condition will be triggered;
  * *negative_overload_threshold* Float: when the overloat exceeds this value for at least 'time_window' seconds, this termination condition is triggered;
  * *big_phi_threshold* Float (deg): when the roll angle exceeds this value for at least 'time_window' seconds, this termination condition is triggered;
  * *is_termination_reward_based_on_steps_left* Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * *termination_reward* Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
