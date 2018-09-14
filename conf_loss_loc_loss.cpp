template <typename Dtype>
void ComputeConfLoss(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss) {
	// num_preds_per_class值为num_priors，因为这里是每一类一个预测得分。
	// -log(softmax[相应label])，如果有匹配就用真实的label，
	// 否则是background的label。
	// all_conf_loss结果为batch * num_preds
  CHECK_LT(background_label_id, num_classes);
  // CHECK_EQ(num, all_match_indices.size());
  all_conf_loss->clear();
  for (int i = 0; i < num; ++i) {
    vector<float> conf_loss;
    const map<int, vector<int> >& match_indices = all_match_indices[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      // Get the label index.
      int label = background_label_id;
      // 这里match_indices只有一个key为-1，值为Nprior长的vector，对应gt索引。
      // 这里如果匹配了，找到对应的gt的label。
      // 否则就是background的label id，label为0
      for (map<int, vector<int> >::const_iterator it =
           match_indices.begin(); it != match_indices.end(); ++it) {
        const vector<int>& match_index = it->second;
        CHECK_EQ(match_index.size(), num_preds_per_class);
        if (match_index[p] > -1) {
          CHECK(all_gt_bboxes.find(i) != all_gt_bboxes.end());
          // 这个图的gt
          const vector<NormalizedBBox>& gt_bboxes =
              all_gt_bboxes.find(i)->second;
          CHECK_LT(match_index[p], gt_bboxes.size());
          // 相应的label
          label = gt_bboxes[match_index[p]].label();
          CHECK_GE(label, 0);
          CHECK_NE(label, background_label_id);
          CHECK_LT(label, num_classes);
          // A prior can only be matched to one gt bbox.
          break;
        }
      }
      Dtype loss = 0;
      if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        CHECK_GE(label, 0);
        CHECK_LT(label, num_classes);
        // Compute softmax probability.
        // We need to subtract the max to avoid numerical issues.
        Dtype maxval = conf_data[start_idx];
        for (int c = 1; c < num_classes; ++c) {
          maxval = std::max<Dtype>(conf_data[start_idx + c], maxval);
        }
        Dtype sum = 0.;
        for (int c = 0; c < num_classes; ++c) {
          sum += std::exp(conf_data[start_idx + c] - maxval);
        }
        Dtype prob = std::exp(conf_data[start_idx + label] - maxval) / sum;
        loss = -log(std::max(prob, Dtype(FLT_MIN)));
      } else if (loss_type == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        int target = 0;
        for (int c = 0; c < num_classes; ++c) {
          if (c == label) {
            target = 1;
          } else {
            target = 0;
          }
          Dtype input = conf_data[start_idx + c];
          loss -= input * (target - (input >= 0)) -
              log(1 + exp(input - 2 * input * (input >= 0)));
        }
      } else {
        LOG(FATAL) << "Unknown conf loss type.";
      }
      conf_loss.push_back(loss);
    }
    conf_data += num_preds_per_class * num_classes;
    all_conf_loss->push_back(conf_loss);
  }
}




template <typename Dtype>
void ComputeLocLoss(const Blob<Dtype>& loc_pred, const Blob<Dtype>& loc_gt,
      const vector<map<int, vector<int> > >& all_match_indices,
      const int num, const int num_priors, const LocLossType loc_loss_type,
      vector<vector<float> >* all_loc_loss) {
	// loc_pred，loc_gt，修改过的loc_pred和loc_gt。
	// 这里对于没有匹配的并没有修改，所以才需要用all_match_indices
	// 来指示需要使用那些来计算。
	// 这里的loc_pred数量是batch * num_priors
	// 同理这里loc_gt是其对应的编码过的gt，也是这个大小。一一对应
  int loc_count = loc_pred.count();
  CHECK_EQ(loc_count, loc_gt.count());
  Blob<Dtype> diff;
  const Dtype* diff_data = NULL;
  if (loc_count != 0) {
    diff.Reshape(loc_pred.shape());
    caffe_sub(loc_count, loc_pred.cpu_data(), loc_gt.cpu_data(),
              diff.mutable_cpu_data());
    diff_data = diff.cpu_data();
  }
  int count = 0;
  for (int i = 0; i < num; ++i) {
    vector<float> loc_loss(num_priors, 0.f);
    for (map<int, vector<int> >::const_iterator
         it = all_match_indices[i].begin();
         it != all_match_indices[i].end(); ++it) {
    	// 这里各个类别的都计算了，实际上只有一个-1的key
      const vector<int>& match_index = it->second;
      CHECK_EQ(num_priors, match_index.size());
      // 对应匹配的才计算，否则是不计算的。
      for (int j = 0; j < match_index.size(); ++j) {
        if (match_index[j] <= -1) {
          continue;
        }
        Dtype loss = 0;
        for (int k = 0; k < 4; ++k) {
          Dtype val = diff_data[count * 4 + k];
          if (loc_loss_type == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
            Dtype abs_val = fabs(val);
            if (abs_val < 1.) {
              loss += 0.5 * val * val;
            } else {
              loss += abs_val - 0.5;
            }
          } else if (loc_loss_type == MultiBoxLossParameter_LocLossType_L2) {
            loss += 0.5 * val * val;
          } else {
            LOG(FATAL) << "Unknown loc loss type.";
          }
        }
        loc_loss[j] = loss;
        ++count;
      }
    }
    all_loc_loss->push_back(loc_loss);
  }
}
