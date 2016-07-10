# OpenCV_Artificial Neural Networks

CvANN_MLP
參數設定:

//Setup the BPNetwork  
    CvANN_MLP bp;   
    // Set up BPNetwork's parameters  
    CvANN_MLP_TrainParams params;  
    params.train_method=CvANN_MLP_TrainParams::BACKPROP;  
    params.bp_dw_scale=0.1;  
    params.bp_moment_scale=0.1;  
    //params.train_method=CvANN_MLP_TrainParams::RPROP;  
    //params.rp_dw0 = 0.1;   
    //params.rp_dw_plus = 1.2;   
    //params.rp_dw_minus = 0.5;  
    //params.rp_dw_min = FLT_EPSILON;   
    //params.rp_dw_max = 50.;



http://docs.opencv.org/2.4/modules/ml/doc/neural_networks.html
