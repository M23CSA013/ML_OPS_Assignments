sudo docker build . -t svm_tree:v1
sudo docker run -it -v /home/usermanaswi/ML_OPS_Assignments:/newdock/host_volume svm_tree:v1