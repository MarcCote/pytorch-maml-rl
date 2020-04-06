# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import sys
import textwrap
from io import StringIO
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
import gym
from gym.utils import colorize
from gym.envs.registration import register, registry

from textworld import EnvInfos
from textworld.envs.wrappers import Filter, GenericEnvironment, Limit
from textworld.envs.batch import AsyncBatchEnv, SyncBatchEnv

from textworld.gym.envs.utils import shuffled_cycle


class TextworldEnv(gym.Env):
    """ Environment for playing TextWorld games.

    Attributes:
        action_space:
            The action space be used with OpenAI baselines.
            (see :py:class:`textworld.gym.spaces.Word <textworld.gym.spaces.text_spaces.Word>`).
        observation_space:
            The observation space be used with OpenAI baselines
            (see :py:class:`textworld.gym.spaces.Word <textworld.gym.spaces.text_spaces.Word>`).

    """
    metadata = {'render.modes': ['human', 'ansi', 'text']}

    def __init__(self,
                 gamefiles: List[str],
                 request_infos: Optional[EnvInfos] = None,
                 task: Dict = {}) -> None:
        """
        Arguments:
            gamefiles:
                Paths of every game composing the pool (`*.ulx|*.z[1-8]`).
            request_infos:
                For customizing the information returned by this environment
                (see
                :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
                for the list of available information).

                .. warning:: Only supported for TextWorld games (i.e., that have a corresponding `*.json` file).

        """
        self.just_been_reset = False
        self.gamefiles = gamefiles
        self._gamefiles = list(self.gamefiles)  # Used to shuffle the ordering.
        self.request_infos = request_infos or EnvInfos()
        self.request_infos.feedback = True
        self.seed(1234)

        self.env = GenericEnvironment(self.request_infos)
        self.env = Filter(self.env)

        self.action_space = gym.spaces.Discrete(1)  # Dummy space
        self.observation_space = gym.spaces.Discrete(1)  # Dummy space

        self._task = task
        self._gamefile = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """ Set the seed for this environment's random generator(s).

        This environment use a random generator to shuffle the order in which
        the games are played.

        Arguments:
            seed: Number that will be used to seed the random generators.

        Returns:
            All the seeds used to set this environment's random generator(s).
        """

        # We shuffle the order in which the game will be seen.
        self._rng_gamefile = np.random.RandomState(seed)
        self._gamefiles = list(self.gamefiles)  # Soft copy to avoid shuffling original list.
        self._rng_gamefile.shuffle(self._gamefiles)
        return [seed]

    def sample_tasks(self, num_tasks):
        # Check if we need more gamefiles than what remains in self._gamefiles.
        while num_tasks > len(self._gamefiles):  # Not enough gamefiles.
            gamefiles = list(self.gamefiles)  # Soft copy to avoid shuffling original list.
            self._rng_gamefile.shuffle(gamefiles)
            self._gamefiles += gamefiles  # Append shuffled gamefiles to iterate through.

        #gamefiles = [next(self._gamefiles_iterator) for _ in range(num_tasks)]
        gamefiles = [self._gamefiles.pop() for _ in range(num_tasks)]
        tasks = [{'gamefile': gamefile} for gamefile in gamefiles]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._gamefile = task['gamefile']

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """ Resets the text-based environment.

        Resetting this environment means starting the next game in the pool.

        Returns:
            A tuple (observations, infos) where

            * observation: text observed in the initial state of the game;
            * infos: additional information as requested for the game.
        """
        self.env.load(self._gamefile)
        self.last_command = None
        self.ob, self.infos = self.env.reset()
        self.just_been_reset = True
        return 0

    def step(self, command: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """ Runs a command in each text-based environment of the batch.

        Arguments:
            commands: Text command to send to the game interpreter.

        Returns:
            A tuple (observation, score, done, infos) where

            * observation: text observed in the new state of the game;
            * score: total number of points accumulated so far in the game;
            * done: whether the game is finished or not;
            * infos: additional information as requested about the game.
        """
        if command == "tw-reset":
            # The purpose of this special command is to make available the infos
            # dict that was supposed to be returned by env.reset.
            msg = "This special command is expected to be sent right after env.reset."
            assert self.just_been_reset, msg
            self.just_been_reset = False
            return 0, 0., False, self.infos

        self.last_command = command
        self.ob, score, done, infos = self.env.step(self.last_command)
        return 0, score, done, infos

    def close(self) -> None:
        """ Close this environment. """

        if self.env is not None:
            self.env.close()

        self.env = None

    # def render(self, mode: str = 'human') -> Optional[Union[StringIO, str]]:
    #     """ Renders the current state of each environment in the batch.

    #     Each rendering is composed of the previous text command (if there's one) and
    #     the text describing the current observation.

    #     Arguments:
    #         mode:
    #             Controls where and how the text is rendered. Supported modes are:

    #                 * human: Display text to the current display or terminal and
    #                   return nothing.
    #                 * ansi: Return a `StringIO` containing a terminal-style
    #                   text representation. The text can include newlines and ANSI
    #                   escape sequences (e.g. for colors).
    #                 * text: Return a string (`str`) containing the text without
    #                   any ANSI escape sequences.

    #     Returns:
    #         Depending on the `mode`, this method returns either nothing, a
    #         string, or a `StringIO` object.
    #     """
    #     outfile = StringIO() if mode in ['ansi', "text"] else sys.stdout

    #     renderings = []
    #     for last_command, ob in zip(self.last_commands, self.obs):
    #         msg = ob.rstrip() + "\n"
    #         if last_command is not None:
    #             command = "> " + last_command
    #             if mode in ["ansi", "human"]:
    #                 command = colorize(command, "yellow", highlight=False)

    #             msg = command + "\n" + msg

    #         if mode == "human":
    #             # Wrap each paragraph at 80 characters.
    #             paragraphs = msg.split("\n")
    #             paragraphs = ["\n".join(textwrap.wrap(paragraph, width=80)) for paragraph in paragraphs]
    #             msg = "\n".join(paragraphs)

    #         renderings.append(msg)

    #     outfile.write("\n-----\n".join(renderings) + "\n")

    #     if mode == "text":
    #         outfile.seek(0)
    #         return outfile.read()

    #     if mode == 'ansi':
    #         return outfile


def register_games(gamefiles: List[str],
                   request_infos: Optional[EnvInfos] = None,
                   name: str = "",
                   **kwargs) -> str:
    """ Make an environment that will cycle through a list of games.

    Arguments:
        gamefiles:
            Paths for the TextWorld games (`*.ulx|*.z[1-8]`).
        request_infos:
            For customizing the information returned by this environment
            (see
            :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
            for the list of available information).

            .. warning:: Only supported for TextWorld games (i.e., with a corresponding `*.json` file).
        name:
            Name for the new environment, i.e. "tw-{name}-v0". By default,
            the returned env_id is "tw-v0".

    Returns:
        The corresponding gym-compatible env_id to use.

    """
    env_id = "tw-{}-v0".format(name) if name else "tw-v0"

    # If env already registered, bump the version number.
    if env_id in registry.env_specs:
        base, _ = env_id.rsplit("-v", 1)
        versions = [int(env_id.rsplit("-v", 1)[-1]) for env_id in registry.env_specs if env_id.startswith(base)]
        env_id = "{}-v{}".format(base, max(versions) + 1)

    entry_point = "maml_rl.envs.textworld:TextworldEnv"

    register(
        id=env_id,
        entry_point=entry_point,
        max_episode_steps=10,
        kwargs={
            'gamefiles': gamefiles,
            'request_infos': request_infos,
            **kwargs}
    )
    return env_id
