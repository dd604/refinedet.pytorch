template <typename Dtype>
void MineHardExamples(const Blob<Dtype>& conf_blob,
    const vector<LabelBBox>& all_loc_preds,
    const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    // all_match_indices会被修改
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices,
	const Dtype* arm_conf_data) {
	// 返回匹配的个数，负样本的个数。
  int num = all_loc_preds.size();
  // CHECK_EQ(num, all_match_overlaps.size());
  // CHECK_EQ(num, all_match_indices->size());
  // all_neg_indices->clear();
  *num_matches = CountNumMatches(*all_match_indices, num);
  *num_negs = 0;
  int num_priors = prior_bboxes.size();
  CHECK_EQ(num_priors, prior_variances.size());
  // Get parameters.
  float objectness_score = multibox_loss_param.objectness_score();
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const int background_label_id = multibox_loss_param.background_label_id();
  const bool use_prior_for_nms = multibox_loss_param.use_prior_for_nms();
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  const MiningType mining_type = multibox_loss_param.mining_type();
  if (mining_type == MultiBoxLossParameter_MiningType_NONE) {
    return;
  }
  const LocLossType loc_loss_type = multibox_loss_param.loc_loss_type();
  const float neg_pos_ratio = multibox_loss_param.neg_pos_ratio();
  const float neg_overlap = multibox_loss_param.neg_overlap();
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool has_nms_param = multibox_loss_param.has_nms_param();
  float nms_threshold = 0;
  int top_k = -1;
  const int sample_size = multibox_loss_param.sample_size();
  // Compute confidence losses based on matching results.
  // 引用all_match_indices，结果放在all_conf_loss, N * prior_num中
  // all_conf_loss结果为batch * num_preds
  // 所有的都有这个损失，positive的是自己的label，negative的则是background的
  vector<vector<float> > all_conf_loss;
  ComputeConfLoss(conf_blob, num, num_priors, num_classes,
      background_label_id, conf_loss_type, *all_match_indices, all_gt_bboxes,
      &all_conf_loss);

  vector<vector<float> > all_loc_loss;
  if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
    // Compute localization losses based on matching results.
    Blob<Dtype> loc_pred, loc_gt;
    if (*num_matches != 0) {
      vector<int> loc_shape(2, 1);
      loc_shape[1] = *num_matches * 4;
      // (1, num_matches * 4)
      // cross batch，同一个batch放在一起进行挖掘
      loc_pred.Reshape(loc_shape);
      loc_gt.Reshape(loc_shape);
      Dtype* loc_pred_data = loc_pred.mutable_cpu_data();
      Dtype* loc_gt_data = loc_gt.mutable_cpu_data();
      // 使用prior_bboxes对相应的gt进行编码，编码结果放在loc_gt_data中
      // 同时挑选相应的all_loc_preds放入到相应的loc_pred中，用于计算loss
      // 没有修改过的loc_pred位置也是有值的，只是我们不需要，所以计算的时候
      // 用all_match_indices，进行指示。
      EncodeLocPrediction(all_loc_preds, all_gt_bboxes, *all_match_indices,
                          prior_bboxes, prior_variances, multibox_loss_param,
                          loc_pred_data, loc_gt_data);
    }
    // batch * num_priors的loss数目，没有匹配的不考虑，匹配的计算smooth_l1
    // 只有正样本才有loc的损失
    ComputeLocLoss(loc_pred, loc_gt, *all_match_indices, num,
                   num_priors, loc_loss_type, &all_loc_loss);
  }
  // 每一张图片，自己单独处理排序排序，记录neg_indices
  // all_neg_indices->push_back(neg_indices);
  // 所有的neg统计数目为num_negs
  // 更新num_matches，num_matches，是表示一个batch内正样本的总数
  // num_negs则是一个batch内负样本的总数
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> >& match_indices = (*all_match_indices)[i];
    const map<int, vector<float> >& match_overlaps = all_match_overlaps[i];
    // loc + conf loss.
    const vector<float>& conf_loss = all_conf_loss[i];
    const vector<float>& loc_loss = all_loc_loss[i];
    vector<float> loss;
    // 将conf_loss和loc_loss相加，放入到loss中。
    std::transform(conf_loss.begin(), conf_loss.end(), loc_loss.begin(),
                   std::back_inserter(loss), std::plus<float>());
    // Pick negatives or hard examples based on loss.
    // 这里是使用阈值过滤掉一部分，并修改索引。
    set<int> sel_indices;
    vector<int> neg_indices;
    for (map<int, vector<int> >::iterator it = match_indices.begin();
         it != match_indices.end(); ++it) {
      const int label = it->first;
      // 这里的it对于目前之后一个，key为-1
      // 所有prior，使用arm得分过滤一下，挑选的(loss, prior_idx)保存到loss_indices中
      int num_sel = 0;
      // Get potential indices and loss pairs.
      vector<pair<float, int> > loss_indices;
      // 这里还都是所有的prior，对于第m个prior，如果arm的conf低，那么丢弃。
      // 否则选择上，记录它的总的loss。
      for (int m = 0; m < match_indices[label].size(); ++m) {
        //对于hardnegative来说这个函数始终返回true
        // 如果没有arm_conf_data，那么就不过滤，否则就用arm_conf进行过滤
        if (IsEligibleMining(mining_type, match_indices[label][m],
            match_overlaps.find(label)->second[m], neg_overlap)) {
          {
            if(arm_conf_data[i*num_priors*2+2*m+1] >= objectness_score){
              loss_indices.push_back(std::make_pair(loss[m], m));
              ++num_sel;
        	}
          }
        }
      }
      if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
        CHECK_GT(sample_size, 0);
        num_sel = std::min(sample_size, num_sel);
      }
      // Select samples.
      {
        // Pick top example indices based on loss.
        // 默认是降序，挑选loss比较高的hard
        std::sort(loss_indices.begin(), loss_indices.end(),
                  SortScorePairDescend<int>);
        for (int n = 0; n < num_sel; ++n) {
          sel_indices.insert(loss_indices[n].second);
        }
      }
      // Update the match_indices and select neg_indices.
      // 更新正样本的匹配的索引match_indices，挑选负样本neg_indices
      for (int m = 0; m < match_indices[label].size(); ++m) {
        // 如果匹配到了，并且没有被select到，
        // 更改索引为-1，表示取消匹配，不考虑，这些本来是
        // 正样本
        // 否则原本就是负样本，并且被选择到了，那么假如到neg_indeces中去。
        if (match_indices[label][m] > -1) {
          if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE &&
              sel_indices.find(m) == sel_indices.end()) {
            match_indices[label][m] = -1;
            *num_matches -= 1;
          }
        } else if (match_indices[label][m] == -1) {
          if (sel_indices.find(m) != sel_indices.end()) {
            neg_indices.push_back(m);
            *num_negs += 1;
          }
        }
      }
    }
    all_neg_indices->push_back(neg_indices);
  }
}
