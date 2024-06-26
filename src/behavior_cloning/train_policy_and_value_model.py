import torch.nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src.common.encoder_resnet_model import EncoderResnetModel
from src.behavior_cloning.image_dataset import ImageDS
from src.common.image_folders import synthetic_training_folders, synthetic_validation_folders


def calculate_accuracy(pred, true_label):
    return torch.sum(torch.max(pred, dim=1).indices == true_label).item() / pred.shape[0]


###########################
# TRAIN POLICY MODEL
###########################

batch_size = 16
batch_size_test_data = 16
policy_model = EncoderResnetModel(5).cuda()
value_model = EncoderResnetModel(1, torch.nn.Sigmoid()).cuda()

policy_criterion = torch.nn.CrossEntropyLoss()
value_criterion = torch.nn.MSELoss()
policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)
value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-5)
policy_scheduler = lr_scheduler.ExponentialLR(policy_optimizer, gamma=0.9998)
value_scheduler = lr_scheduler.ExponentialLR(value_optimizer, gamma=0.9998)

ds = ImageDS(synthetic_training_folders, 'png')
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

test_ds = ImageDS(synthetic_validation_folders, 'png')
test_dataloader = DataLoader(test_ds, batch_size=batch_size_test_data, shuffle=True, num_workers=0)

train_losses_policy = []
test_losses_policy = []
train_losses_value = []
test_losses_value = []
test_accuracies_policy = []
train_accuracies_policy = []

def save_losses():
    file = open("res/behavior_cloning/training_loss_policy_yt.pt", "w")
    file.write(repr(train_losses_policy))
    file.close()

    file = open("res/behavior_cloning/validation_loss_policy_yt.pt", "w")
    file.write(repr(test_losses_policy))
    file.close()

    file = open("res/behavior_cloning/validation_acc_yt.pt", "w")
    file.write(repr(test_accuracies_policy))
    file.close()

    file = open("res/behavior_cloning/train_acc_yt.pt", "w")
    file.write(repr(train_accuracies_policy))
    file.close()

    file = open("res/behavior_cloning/validation_loss_value_yt.pt", "w")
    file.write(repr(test_losses_value))
    file.close()
    file = open("res/behavior_cloning/training_loss_value_yt.pt", "w")
    file.write(repr(train_losses_value))
    file.close()


print(f'Step | Train loss | Train Acc | Test loss | Accuracy | Train value loss | Test value loss')
mean_loss = 0
mean_value_loss = 0
train_policy_model = True
train_value_model = True
for step in range(0, 250000):
    ds_iterator = iter(dataloader)
    start_image, current_image, _, action, value = next(ds_iterator)

    if train_policy_model:
        policy_model.train()

        predicted_action = policy_model(start_image, current_image)
        loss = policy_criterion(predicted_action, action)

        mean_loss += loss.item()

        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()
        if policy_optimizer.param_groups[0]['lr'] > 1e-6:
            policy_scheduler.step()

    if train_value_model:
        value_model.train()

        predicted_value = value_model(start_image, current_image)
        loss = value_criterion(predicted_value, value.unsqueeze(-1))

        mean_value_loss += loss.item()

        value_optimizer.zero_grad()
        loss.backward()
        value_optimizer.step()
        if value_optimizer.param_groups[0]['lr'] > 1e-6:
            value_scheduler.step()

    if step % 500 == 0 and step != 0:
        with torch.no_grad():
            test_loss = 0
            test_value_loss = 0
            test_acc_temp = 0
            train_acc_temp = 0
            train_loss = 0
            train_value_loss = 0
            rounds = 10
            for _ in range(rounds):
                start_image, current_image, _, action, value = next(ds_iterator)
                predicted_action = policy_model(start_image, current_image)
                predicted_value = value_model(start_image, current_image)
                train_acc_temp += calculate_accuracy(predicted_action, action)
                train_loss += policy_criterion(predicted_action, action).item()
                train_value_loss += value_criterion(predicted_value, value.unsqueeze(-1)).item()

                test_ds_iterator = iter(test_dataloader)
                start_image, current_image, _, action, value = next(test_ds_iterator)
                predicted_action = policy_model(start_image, current_image)
                predicted_value = value_model(start_image, current_image)
                test_loss += policy_criterion(predicted_action, action).item()
                test_value_loss += value_criterion(predicted_value, value.unsqueeze(-1)).item()
                test_acc_temp += calculate_accuracy(predicted_action, action)

            print(f'{step:5d} | ' f'{mean_loss / 500.0:.8f} | {train_acc_temp / 10.0:.2f} | {test_loss / 10.0:.8f} | {test_acc_temp / 10.0:.2f} | \
                {train_value_loss / 10:.8f} | {test_value_loss / 10:.8f}')

            train_losses_policy.append(train_loss / 10.0)
            test_losses_policy.append(test_loss / 10.0)
            train_losses_value.append(train_value_loss / 10.0)
            test_losses_value.append(test_value_loss / 10.0)
            test_accuracies_policy.append(test_acc_temp / 10.0)
            train_accuracies_policy.append(train_acc_temp / 10.0)
            save_losses()

        if step % 5000 == 0 and step > 10000 and (test_acc_temp / 10.0) > 0.6:
            if train_policy_model:
                torch.save(policy_model.state_dict(), f'res/behavior_cloning/for_pdf_policy_yt_{step}.pth')
            if train_value_model:
                torch.save(value_model.state_dict(), f'res/behavior_cloning/for_pdf_value_yt_{step}.pth')
        mean_loss = 0
        mean_value_loss = 0

