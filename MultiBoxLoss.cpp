template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const Dtype* gt_data = bottom[3]->cpu_data();
  const Dtype* arm_conf_data = NULL;
  const Dtype* arm_loc_data = NULL;
  vector<LabelBBox> all_arm_loc_preds;
  if (bottom.size() >= 5) {
	arm_conf_data = bottom[4]->cpu_data();
  }
  if (bottom.size() >= 6) {
	arm_loc_data = bottom[5]->cpu_data();
	GetLocPredictions(arm_loc_data, num_, num_priors_, loc_classes_, share_location_,
	                  &all_arm_loc_preds);
  }

  // Retrieve all ground truth.
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_, num_classes_,
                 &all_gt_bboxes);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Retrieve all predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num_, num_priors_, loc_classes_, share_location_,
                    &all_loc_preds);

  // Find matches between source bboxes and ground truth bboxes.
  vector<map<int, vector<float> > > all_match_overlaps;
  if (bottom.size() >= 6) {
    // 使用prior_bboxes对all_loc_preds进行解码，得到refine_bboxes，
    // 使用refine_bboxes与gt_bboxes进行匹配
    // 获取all_match_indices_
	CasRegFindMatches(all_loc_preds, all_gt_bboxes, prior_bboxes, prior_variances,
			    multibox_loss_param_, &all_match_overlaps, &all_match_indices_,
			    all_arm_loc_preds);
  }
  else {
    FindMatches(all_loc_preds, all_gt_bboxes, prior_bboxes, prior_variances,
                multibox_loss_param_, &all_match_overlaps, &all_match_indices_);
  }

  num_matches_ = 0;
  int num_negs = 0;
  // Sample hard negative (and positive) examples based on mining type.
  // 这里用到EncodeLocPrediction？
  // 这里用的odm的loc_preds，用的是普通的prior_bboxes对gt进行编码，
  // 并没有用到arm的loc，只是用到了arm的conf，而是使用了refine得到的索引进行监督。
  // 很是奇怪。
  // 会修改all_match_indices_
  MineHardExamples(*bottom[1], all_loc_preds, all_gt_bboxes, prior_bboxes,
                   prior_variances, all_match_overlaps, multibox_loss_param_,
                   &num_matches_, &num_negs, &all_match_indices_,
                   &all_neg_indices_, arm_conf_data);
  // 如果有匹配，再进行一次解码，不进行匹配了，使用上面的all_match_indices_
  // 得到loc_pred_和loc_gt_，用来计算loss的对。
  // 这里CasRegEncodeLocPrediction
  if (num_matches_ >= 1) {
    // Form data to pass on to loc_loss_layer_.
    vector<int> loc_shape(2);
    loc_shape[0] = 1;
    loc_shape[1] = num_matches_ * 4;
    loc_pred_.Reshape(loc_shape);
    loc_gt_.Reshape(loc_shape);
    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
    // 不会修改all_match_indices_
    CasRegEncodeLocPrediction(all_loc_preds, all_gt_bboxes,
                    all_match_indices_,
                    prior_bboxes, prior_variances, multibox_loss_param_,
                    loc_pred_data, loc_gt_data, all_arm_loc_preds);
    loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
  }

  // Form data to pass on to conf_loss_layer_.
  if (do_neg_mining_) {
    num_conf_ = num_matches_ + num_negs;
  } else {
    num_conf_ = num_ * num_priors_;
  }
  if (num_conf_ >= 1) {
    // Reshape the confidence data.
    vector<int> conf_shape;
    if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      conf_shape.push_back(num_conf_);
      conf_gt_.Reshape(conf_shape);
      conf_shape.push_back(num_classes_);
      conf_pred_.Reshape(conf_shape);
    } else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      conf_shape.push_back(1);
      conf_shape.push_back(num_conf_);
      conf_shape.push_back(num_classes_);
      conf_gt_.Reshape(conf_shape);
      conf_pred_.Reshape(conf_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      // Share data and diff with bottom[1].
      CHECK_EQ(conf_pred_.count(), bottom[1]->count());
      conf_pred_.ShareData(*(bottom[1]));
    }
    // 抽取了正负样本数量之后，这里是抽取的有效的部分，进行conf的编码
    Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
    caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
    EncodeConfPrediction(conf_data, num_, num_priors_, multibox_loss_param_,
                         all_match_indices_, all_neg_indices_, all_gt_bboxes,
                         conf_pred_data, conf_gt_data);
    conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
    conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
  } else {
    conf_loss_.mutable_cpu_data()[0] = 0;
  }

  top[0]->mutable_cpu_data()[0] = 0;
  if (this->layer_param_.propagate_down(0)) {
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
    top[0]->mutable_cpu_data()[0] +=
        loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
  }
  if (this->layer_param_.propagate_down(1)) {
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
    top[0]->mutable_cpu_data()[0] += conf_loss_.cpu_data()[0] / normalizer;
  }
}
