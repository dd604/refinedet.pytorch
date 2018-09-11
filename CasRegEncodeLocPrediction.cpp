template <typename Dtype>
void CasRegEncodeLocPrediction(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      Dtype* loc_pred_data, Dtype* loc_gt_data,
	  const vector<LabelBBox>& all_arm_loc_preds) {
	  // 进行encode，结果保存在Dtype* loc_pred_data, Dtype* loc_gt_data,中
  // 图片数目
  int num = all_loc_preds.size();
  // CHECK_EQ(num, all_match_indices.size());
  // Get parameters.
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  // 是否会用到在中间的判断
  const bool bp_inside = multibox_loss_param.bp_inside();
  const bool use_prior_for_matching =
      multibox_loss_param.use_prior_for_matching();
  int count = 0;
  for (int i = 0; i < num; ++i) {
    //apply arm_loc_preds to prior_box
    // decode arm_loc_preds by prior_box
    // i图，当前图arm预测
    const vector<NormalizedBBox>& arm_loc_preds =
        all_arm_loc_preds[i].find(-1)->second;
    // 存放解码结果，又解一次码？得到的是refine priors，
    // 和前面寻找match是一样的方式。
    vector<NormalizedBBox> decode_prior_bboxes;
    bool clip_bbox = false;
    DecodeBBoxes(prior_bboxes, prior_variances,
    		code_type, encode_variance_in_target, clip_bbox,
			arm_loc_preds, &decode_prior_bboxes);
    // indices guide matching between all_loc_preds and gt.
    // 在前面find match基础上进一步进行匹配arm匹配的refine box，对应的进行encode。
    // 当前图arm的匹配索引。
    for (map<int, vector<int> >::const_iterator
         it = all_match_indices[i].begin();
         it != all_match_indices[i].end(); ++it) {
      // 当前类别class
      const int label = it->first;
      const vector<int>& match_index = it->second;
      // 一定要能保证存在
      CHECK(all_loc_preds[i].find(label) != all_loc_preds[i].end());
      // 找到匹配的预测，这里是要拷贝这个值放到另外的loc_pred_data里面。
      // 主要只是抽取作用。
      const vector<NormalizedBBox>& loc_pred =
          all_loc_preds[i].find(label)->second;
      for (int j = 0; j < match_index.size(); ++j) {
        if (match_index[j] <= -1) {
          continue;
        }
        // Store encoded ground truth.
        const int gt_idx = match_index[j];
        CHECK(all_gt_bboxes.find(i) != all_gt_bboxes.end());
        CHECK_LT(gt_idx, all_gt_bboxes.find(i)->second.size());
        const NormalizedBBox& gt_bbox = all_gt_bboxes.find(i)->second[gt_idx];
        NormalizedBBox gt_encode;
        CHECK_LT(j, decode_prior_bboxes.size());
        // 找到对应的gt，用对应的refine过的prior对gt进行编码，得到loc_gt_data
        EncodeBBox(decode_prior_bboxes[j], prior_variances[j], code_type,
                   encode_variance_in_target, gt_bbox, &gt_encode);
        // 保存gt进行编码的结果
        loc_gt_data[count * 4] = gt_encode.xmin();
        loc_gt_data[count * 4 + 1] = gt_encode.ymin();
        loc_gt_data[count * 4 + 2] = gt_encode.xmax();
        loc_gt_data[count * 4 + 3] = gt_encode.ymax();
        // Store location prediction.
        CHECK_LT(j, loc_pred.size());
        // 保存预测的结果，后面二者计算loss
        {
          loc_pred_data[count * 4] = loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] = loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] = loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] = loc_pred[j].ymax();
        }
        if (encode_variance_in_target) {
          for (int k = 0; k < 4; ++k) {
            CHECK_GT(prior_variances[j][k], 0);
            loc_pred_data[count * 4 + k] /= prior_variances[j][k];
            loc_gt_data[count * 4 + k] /= prior_variances[j][k];
          }
        }
        ++count;
      }
    }
  }
}