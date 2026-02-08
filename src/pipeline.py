import importlib
import argparse
import inspect
import itertools
import time
import numpy as np
import logging
import toml
import yaml

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from .utils.StateLogger import StateLogger

# Set the basic config for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Формат даты
)


class Pipeline:
    def __init__(self, parsed_dict=None):
        self.args_dict = None
        self.set_env(parsed_dict)

    def _import_class(self, folder_name, class_name):
        try:
            module_path = "src." + folder_name + "." + class_name
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                raise ModuleNotFoundError(f"No module named {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            class_name = class_name.replace(".py", "")
            cls = getattr(module, class_name)
            return cls
        except Exception as e:
            error_message = f"Ошибка при импорте класса {class_name}: {e}"
            logging.error(error_message)
            raise ValueError(error_message)

    def load_config(self, file_name, config_name):
        with open(file_name, "r", encoding="utf-8") as config_file:
            config = toml.load(config_file)
        return {
            "drone": DroneModel(config[config_name]["drone"]),
            "num_drones": config[config_name]["num_drones"],
            "gui": config[config_name]["gui"],
            "record_video": config[config_name]["record_video"],
            "vision_attributes": config[config_name]["vision_attributes"],
            "save_images": config[config_name]["save_images"],
            "save_video": config[config_name]["save_video"],
            "simulation_freq_hz": config[config_name]["simulation_freq_hz"],
            "control_freq_hz": config[config_name]["control_freq_hz"],
            "duration_sec": config[config_name]["duration_sec"],
            "output_folder": config[config_name]["output_folder"],
            "colab": config[config_name]["colab"],
            "plot": config[config_name]["plot"],
            "test": config[config_name]["test"],
            "env_name": config[config_name]["env_name"],
            "decision_name": config[config_name]["decision_name"],
            "experiment_name": config[config_name]["experiment_name"],
            "infinity": config[config_name]["infinity"],
        }

    def load_models(self, config_name: str, experiment_name: str) -> list:
        with open(config_name, 'r', encoding="utf-8") as config_file:
            models = yaml.safe_load(config_file)

        return models.get(experiment_name, [])

    def set_env(self, test_dict=None):
        parser = argparse.ArgumentParser(
            description="This is the runner for drones autopilot"
        )
        parser.add_argument(
            "--drone",
            default=DroneModel("cf2x"),
            type=DroneModel,
            help="Drone model (default: CF2X)",
            metavar="",
            choices=DroneModel,
        )
        parser.add_argument(
            "--num_drones",
            default=1,
            type=int,
            help="Number of drones (default: 1)",
            metavar="",
        )
        parser.add_argument(
            "--gui",
            default=True,
            type=str2bool,
            help="Whether to use PyBullet GUI (default: True)",
            metavar="",
        )
        parser.add_argument(
            "--record_video",
            default=False,
            type=str2bool,
            help="Whether to record a video (default: False)",
            metavar="",
        )
        parser.add_argument(
            "--vision_attributes",
            default=False,
            type=str2bool,
            help="Whether to record a video frome drone (default: False)",
            metavar="",
        )
        parser.add_argument(
            "--save_images",
            default=False,
            type=str2bool,
            help="Whether to save frames frome drone (default: False)",
            metavar="",
        )
        parser.add_argument(
            "--save_video",
            default=False,
            type=str2bool,
            help="Whether to save video frome drone (default: False)",
            metavar="",
        )
        parser.add_argument(
            "--simulation_freq_hz",
            default=240,
            type=int,
            help="Simulation frequency in Hz (default: 240)",
            metavar="",
        )
        parser.add_argument(
            "--control_freq_hz",
            default=48,
            type=int,
            help="Control frequency in Hz (default: 48)",
            metavar="",
        )
        parser.add_argument(
            "--duration_sec",
            default=12,
            type=int,
            help="Duration of the simulation in seconds (default: 10)",
            metavar="",
        )
        parser.add_argument(
            "--output_folder",
            default="results",
            type=str,
            help='Folder where to save logs (default: "results")',
            metavar="",
        )
        parser.add_argument(
            "--colab",
            default=False,
            type=bool,
            help='Whether example is being run by a notebook (default: "False")',
            metavar="",
        )
        parser.add_argument(
            "--plot",
            default=True,
            type=bool,
            help='Whether example is being run with plots (default: "True")',
            metavar="",
        )
        parser.add_argument(
            "--save_logs",
            default=True,
            type=bool,
            help='Whether example is being run with saving logs (default: "True")',
            metavar="",
        )
        parser.add_argument(
            "--test",
            default=False,
            type=bool,
            help='Whether example is being run with test - exit after 5 steps (default: "False")',
            metavar="",
        )
        parser.add_argument(
            "--env_name",
            default="AutoAviary",
            type=str,
            help="Please note that the environment name should be located in the src/envs folder,"
            " in a file named the same as the environment class it contains. For example,"
            " the AutoAviary environment is located in src/envs/AutoAviary."
            " (default: 'AutoAviary')",
            metavar="",
        )
        parser.add_argument(
            "--decision_name",
            default="Decision",
            type=str,
            help="Please note that the Decision name should be located in the src/experiments"
            " folder, in a file named the same as the decision class it contains."
            " For example, the Decision class is located in src/experiments/Decision."
            " (default: 'Decision')",
            metavar="",
        )
        parser.add_argument(
            "--experiment_name",
            default="default",
            type=str,
            help="Name of the experiment that contains its own set of environment models"
                 " (default: 'default')",
            metavar=""
        )
        parser.add_argument(
            "--infinity",
            default=True,
            type=str2bool,
            help="Whether to set infinity simulation (default: True).",
            metavar="",
        )
        parser.add_argument(
            "--separating_step",
            default=0.4,
            type=float,
            help="drone spawn distance == [drone_i * separating_step]",
            metavar="",
        )
        parser.add_argument(
            "--config", help="Configuration name in the configuration file config.toml"
        )
        parser.add_argument("other_args", nargs="*", help="Ignored arguments")
        args = parser.parse_known_args()[0]

        if args.config is not None:
            try:
                args = self.load_config("config.toml", args.config)
            except KeyError:
                error_message = "ERROR :: In the provided configuration file,"
                "the configuration is either missing or corrupted."
                logging.error(error_message)
                raise error_message
        elif test_dict is not None:
            args = test_dict
        else:
            args = {k: v for k, v in vars(args).items() if v is not None}

        self.args_dict = args
        self.infinity = args["infinity"]
        self.num_drones = args["num_drones"]
        self.separating_step = args["separating_step"]
        self.decision_name = args["decision_name"]
        self.env_name = args["env_name"]
        self.test = args["test"]
        self.plot = args["plot"]
        self.save_logs = args["save_logs"]
        self.vision_attributes = args["vision_attributes"]
        self.save_images = args["save_images"]
        self.gui = args["gui"]
        self.control_freq_hz = args["control_freq_hz"]
        self.models = self.load_models(config_name="objects.yaml",
                                       experiment_name=args["experiment_name"])
        self.init_xyzs = np.array([[i * self.separating_step] * 3 for i in range(self.num_drones)])
        ENV = self._import_class("envs", self.env_name)

        env_args = {
            "drone_model": args["drone"],
            "num_drones": args["num_drones"],
            "gui": args["gui"],
            "vision_attributes": args["vision_attributes"],
            "save_images": args["save_images"],
            "save_video": args["save_video"],
            "output_folder": args["output_folder"],
            "models": self.models,
            "initial_xyzs": self.init_xyzs
        }

        env_init_signature = inspect.signature(ENV.__init__)
        env_init_signature_keys = list(env_init_signature.parameters.keys())
        for arg_name, arg_value in args.items():
            logging.info(f"{arg_name} and {arg_value}")
            if arg_name not in env_args:
                if arg_name in env_init_signature_keys:
                    env_args[arg_name] = arg_value
                else:
                    logging.warning(f"unknown argument ignored: {arg_name}")
        self.env = ENV(**env_args)

    def set_logger(self):
        self.logger = StateLogger(
            logging_freq_hz=self.args_dict["control_freq_hz"],
            num_drones=self.args_dict["num_drones"],
            duration_sec= 0 if self.infinity else self.args_dict["duration_sec"],
            output_folder=self.args_dict["output_folder"],
            colab=self.args_dict["colab"],
        )

    def set_ctrl(self):
        self.ctrl = [DSLPIDControl(drone_model=self.args_dict["drone"]) for _ in range(self.num_drones)]

    def set_desig(self):
        DES = self._import_class("experiments", self.decision_name)
        self.desig = DES(self.ctrl)

    def stop(self):
        self.env.close()

        if self.plot:
            self.logger.plot()
        if self.save_logs:
            self.logger.save_as_csv()

    def run(self):
        try:
            x, y, z = 0, 3, 0
            self.INIT_XYZS = np.array([[x, y, z]])

            self.set_logger()
            self.set_ctrl()
            self.set_desig()
            action = np.zeros((self.num_drones, 4))
            START = time.time()

            iterator = (
                itertools.count()
                if self.infinity
                else range(0, int(self.duration_sec * self.control_freq_hz))
            )

            for i in iterator:
                # ==== Step the simulation =====
                obs, _reward, _terminated, _truncated, _info = self.env.step(action)
                target_poss = []
                target_rpys= []
                target_vels = []
                if self.vision_attributes or self.save_images:
                    self.env.take_image()
                    for j in range(self.num_drones):
                        target_pos, target_rpy, target_vel = self.desig.update_move(
                            iteration=i, image=self.env.rgb, obs=obs, drone_serial_number=j
                        )
                        target_poss.append(target_pos)
                        target_rpys.append(target_rpy)
                        target_vels.append(target_vel)
                else:
                    for j in range(self.num_drones):
                        target_pos, target_rpy, target_vel = self.desig.update_move(iteration=i, obs=obs, drone_serial_number=j)
                        target_poss.append(target_pos)
                        target_rpys.append(target_rpy)
                        target_vels.append(target_vel)
                
                for j in range(self.num_drones):
                    action[j], _, _ = self.ctrl[j].computeControlFromState(
                        control_timestep=self.env.CTRL_TIMESTEP,
                        state=obs[j],
                        target_pos=target_poss[j],
                        target_rpy=target_rpys[j],
                        target_vel=target_vels[j],
                    )

                for j in range(self.num_drones):
                    target_data = np.concatenate((target_poss[j], target_vels[j], target_rpys[j], [0,0,0])) 
                    self.logger.log(drone=j, timestamp=i / self.env.CTRL_FREQ, state=obs[j], control = target_data)
                self.env.render()
                if self.gui:
                    sync(i, START, self.env.CTRL_TIMESTEP)
            self.stop()
        except KeyboardInterrupt:
            print(f"\nSystem interrupt received\n")
            self.stop()

        except Exception as e:
            print(f"The program terminates with the message: {e}")
            self.stop()
            try:
                self.stop()
            except Exception as e:
                print(f"Failed to plot graphs or save data. Error: {e}")

