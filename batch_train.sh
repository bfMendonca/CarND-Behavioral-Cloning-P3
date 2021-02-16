#!/bin/bash
python model.py NVIDIANetV0 && for i in {1..9}; do python retrain.py NVIDIANetV3; done
python model.py NVIDIANetV1 && for i in {1..9}; do python retrain.py NVIDIANetV3; done
python model.py NVIDIANetV2 && for i in {1..9}; do python retrain.py NVIDIANetV3; done
python model.py NVIDIANetV3 && for i in {1..9}; do python retrain.py NVIDIANetV3; done
python model.py NVIDIANetV4 && for i in {1..9}; do python retrain.py NVIDIANetV4; done
python model.py NVIDIANetV5 && for i in {1..9}; do python retrain.py NVIDIANetV5; done
python model.py NVIDIANetV6 && for i in {1..9}; do python retrain.py NVIDIANetV6; done
python model.py NVIDIANetV6 && for i in {1..9}; do python retrain.py NVIDIANetV6; done

python model.py ModNVIDIANetV1 && for i in {1..9}; do python retrain.py ModNVIDIANetV1; done
python model.py ModNVIDIANetV2 && for i in {1..9}; do python retrain.py ModNVIDIANetV2; done
python model.py ModNVIDIANetV3 && for i in {1..9}; do python retrain.py ModNVIDIANetV3; done
