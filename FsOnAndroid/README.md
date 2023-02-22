# FS-Real Android Runtime

## Introduction

This repository contains the android runtime for FS-Real. You can build and install the APP to 
- participate federated training, 
- monitor the federated/local training progress,
- monitor the device status (e.g. memory and cpu usage), and 
- adjust hyper-parameters.

## Guidance

The training configuration is loaded/stored via YAML. The options contain (not limited to) the following content: 
```yaml
data:
  batch_size:         64
  root:               /data/user/0/com.example.fsandroid/files/
  type:               MNIST
distribute:
  device_port:        50078
  report_host:        0.0.0.0
  report_port:        50078
  server_host:        8.210.21.85
  server_port:        50051
model: 
  type:               LeNet
optimizer:
  lr:                 0.1
  momentum:           0.0
  weight_decay:       0.0
task:
  criterion:          CrossEntropyLoss
  type:               classification
train:
  auto_start:         true
  local_update_steps: 1
```
You can load your customized configuration by placing a yaml file named "config.yaml" in `/data/user/0/com.example.fsandroid/files/`. 
The APP will load it and cover the default configurations automatically. 
Also, you can set the configuration by clicking the setting button. 
You can customize your own config in the directory `app/src/main/java/com/example/fsandroid/configs`.

## Notice

For now, to receive messages from server, we run a grpc server on android device, which means the IP address of android device should be stable.
You can achieve it via LAN, or port mapping. 
