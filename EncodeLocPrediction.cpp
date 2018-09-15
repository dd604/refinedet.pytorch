template <typename Dtype>
void EncodeConfPrediction(const Dtype* conf_data, const int num,
      const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<vector<int> >& all_neg_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      Dtype* conf_pred_data, Dtype* conf_gt_data) {
  // CHECK_EQ(num, all_match_indices.size());
  // CHECK_EQ(num, all_neg_indices.size());
  // Retrieve parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const int background_label_id = multibox_loss_param.background_label_id();
  const bool map_object_to_agnostic =
      multibox_loss_param.map_object_to_agnostic();
  if (map_object_to_agnostic) {
    if (background_label_id >= 0) {
      CHECK_EQ(num_classes, 2);
    } else {
      CHECK_EQ(num_classes, 1);
    }
  }
  const MiningType mining_type = multibox_loss_param.mining_type();
  bool do_neg_mining;
  if (multibox_loss_param.has_do_neg_mining()) {
    LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
    do_neg_mining = multibox_loss_param.do_neg_mining();
    CHECK_EQ(do_neg_mining,
             mining_type != MultiBoxLossParameter_MiningType_NONE);
  }
  do_neg_mining = mining_type != MultiBoxLossParameter_MiningType_NONE;
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  int count = 0;
  for (int i = 0; i < num; ++i) {
  	// 如果这个图像存在gt bboxes
    if (all_gt_bboxes.find(i) != all_gt_bboxes.end()) {
      // Save matched (positive) bboxes scores and labels.
      const map<int, vector<int> >& match_indices = all_match_indices[i];
      for (map<int, vector<int> >::const_iterator it =
          match_indices.begin(); it != match_indices.end(); ++it) {
        const vector<int>& match_index = it->second;
        CHECK_EQ(match_index.size(), num_priors);
        for (int j = 0; j < num_priors; ++j) {
          if (match_index[j] <= -1) {
            continue;
          }
          // gt label,这里是gt，softmax时候的conf_gt
          // 这次是抽取对应的pred，conf_pred_data
          const int gt_label = map_object_to_agnostic ?
            background_label_id + 1 :
            all_gt_bboxes.find(i)->second[match_index[j]].label();
          int idx = do_neg_mining ? count : j;
          switch (conf_loss_type) {
            case MultiBoxLossParameter_ConfLossType_SOFTMAX:
              conf_gt_data[idx] = gt_label;
              break;
            case MultiBoxLossParameter_ConfLossType_LOGISTIC:
              conf_gt_data[idx * num_classes + gt_label] = 1;
              break;
            default:
              LOG(FATAL) << "Unknown conf loss type.";
          }
          if (do_neg_mining) {
            // Copy scores for matched bboxes.
            caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
                conf_pred_data + count * num_classes);
            ++count;
          }
        }
      }
      // Go to next image.
      if (do_neg_mining) {
        // Save negative bboxes scores and labels.
        for (int n = 0; n < all_neg_indices[i].size(); ++n) {
          int j = all_neg_indices[i][n];
          CHECK_LT(j, num_priors);
          caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
              conf_pred_data + count * num_classes);
          switch (conf_loss_type) {
            case MultiBoxLossParameter_ConfLossType_SOFTMAX:
              conf_gt_data[count] = background_label_id;
              break;
            case MultiBoxLossParameter_ConfLossType_LOGISTIC:
              if (background_label_id >= 0 &&
                  background_label_id < num_classes) {
                conf_gt_data[count * num_classes + background_label_id] = 1;
              }
              break;
            default:
              LOG(FATAL) << "Unknown conf loss type.";
          }
          ++count;
        }
      }
    }
    if (do_neg_mining) {
      conf_data += num_priors * num_classes;
    } else {
      conf_gt_data += num_priors;
    }
  }
}
