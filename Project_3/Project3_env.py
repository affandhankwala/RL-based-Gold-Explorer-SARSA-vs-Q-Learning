# Corrected version 2, 7 March 2024

import random


def get_state_index(x, y, z):
    x_idx = x
    y_idx = 8 * y
    z_idx = 64 * z
    return x_idx + y_idx + z_idx   # ranges from 0 to 127


class GoldExplorer:
    """The Gold Explorer puzzle"""

    def __init__(self):

        self.num_states = 128
        self.num_actions = 4
        self.expl_x = 0  # explorer's x position from 0 to 7
        self.expl_y = 0  # explorer's y position from 0 to 7
        self.expl_z = 0  # explorer's z position from 0 to 1
        self.win = {15, 62, 79, 126}
        self.loss = {13, 17, 28, 32, 51, 77, 81, 92, 96, 115}
        self.coins = {50}
        self.mount = {1, 9, 49, 65, 73, 113}

    # Get the key environment parameters
    def get_number_of_states(self):
        return self.num_states

    def get_number_of_actions(self):
        return self.num_actions

    # Get the state IDs that should not be set optimistically
    def get_terminal_states(self):
        term = self.win.union(self.loss, self.mount)
        return term

    def get_state(self):
        return get_state_index(self.expl_x, self.expl_y, self.expl_z)

    # Set the current state to the initial state
    def reset(self, exp_starts):
        x = 0
        y = 0
        z = 0
        if exp_starts:
            done = False
            while not done:
                x = random.randint(0, 7)
                y = random.randint(0, 7)
                z = random.randint(0, 1)
                st = get_state_index(x, y, z)
                if (st in self.win) or (st in self.loss) or (st in self.mount) or (st in self.coins):
                    done = False
                else:
                    done = True
        self.expl_x = x
        self.expl_y = y
        self.expl_z = z
        st = get_state_index(self.expl_x, self.expl_y, self.expl_z)
        return st

    def execute_action(self, action):
        # Use the agent's action to determine the next state and reward #
        # Note: 'N' = 0, 'E' = 1, 'S' = 2, 'W' = 3 #

        current_state = get_state_index(self.expl_x, self.expl_y, self.expl_z)
        new_state = current_state
        reward = 0
        game_end = False

        # if in terminal states, stay in terminal states
        if (current_state in self.win) or (current_state in self.loss):
            new_state = current_state
            reward = 0
            game_end = True

        elif (current_state in self.mount) or (current_state in self.coins):
            new_state = current_state
            reward = -1000
            game_end = True

        else:
            temp_x = self.expl_x
            temp_y = self.expl_y
            temp_z = self.expl_z

            # determine a potential next state
            if action == 0:  # action is 'N'
                if temp_y == 0:
                    temp_y = 0
                else:
                    temp_y = temp_y - 1

            elif action == 1:  # action is 'E'
                if temp_x == 7:
                    temp_x = 7
                else:
                    temp_x = temp_x + 1

            elif action == 2:  # action is 'S'
                if temp_y == 7:
                    temp_y = 7
                else:
                    temp_y = temp_y + 1

            else:  # action is 'W'
                if temp_x == 0:
                    temp_x = 0
                else:
                    temp_x = temp_x - 1

            # recalculate the new state
            new_state = get_state_index(temp_x, temp_y, temp_z)

            # check to see if coins can be picked up
            if new_state in self.coins:
                temp_z = 1  # shift to second level grid space
                new_state = get_state_index(temp_x, temp_y, temp_z)
                reward = 15
                game_end = False

            elif new_state in self.mount:
                temp_x = self.expl_x
                temp_y = self.expl_y
                temp_z = self.expl_z
                new_state = get_state_index(temp_x, temp_y, temp_z)
                reward = -1
                game_end = False

            elif new_state in self.loss:      # you lose
                reward = -30
                game_end = True

            elif new_state in self.win:     # you won
                reward = 30
                game_end = True

            else:
                reward = -1
                game_end = False

            self.expl_x = temp_x
            self.expl_y = temp_y
            self.expl_z = temp_z

        return new_state, reward, game_end



