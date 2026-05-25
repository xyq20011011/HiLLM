from trainer import Trainer
from Models.DeepCD import HiLLM


if __name__ == "__main__":
    model = HiLLM
    dataset_name = "MOOC_CS"
    exp_name = f"{repr(model)}_{dataset_name}"
    print(exp_name)

    trainer = Trainer(exp_name)
    trainer.verbose = False
    trainer.load_data(name=dataset_name)
    trainer.init_model(model, dataset_name=dataset_name)

    trainer.train(to_epoch=100, dataset_name=dataset_name)


