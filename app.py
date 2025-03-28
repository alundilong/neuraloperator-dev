from melting.app.app import create_app
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

if __name__ == "__main__":
    # Read the configuration
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./melting_eval_config.yaml", config_name="default", config_folder="./melting/app/configs"
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder="../config"),
        ]
    )
    config = pipe.read_conf()

    app = create_app(config=config)
    app.launch()
