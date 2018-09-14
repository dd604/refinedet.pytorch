void CasRegFindMatches(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      vector<map<int, vector<float> > >* all_match_overlaps,
      vector<map<int, vector<int> > >* all_match_indices,
	  const vector<LabelBBox>& all_arm_loc_preds) {
  // all_match_overlaps->clear();
  // all_match_indices->clear();
  // Get parameters.
  //all_match_indices第一层vector对应，map的int对应label，这里不区分类别，
  // key为-1 数目是Batch * ([label=-1], Nprior)
  //all_match_overlaps类似
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const bool share_location = multibox_loss_param.share_location();
  // 共享位置，loc_classes = 1
  const int loc_classes = share_location ? 1 : num_classes;
  const MatchType match_type = multibox_loss_param.match_type();
  const float overlap_threshold = multibox_loss_param.overlap_threshold();
  const bool use_prior_for_matching =
      multibox_loss_param.use_prior_for_matching();
  const int background_label_id = multibox_loss_param.background_label_id();
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool ignore_cross_boundary_bbox =
      multibox_loss_param.ignore_cross_boundary_bbox();
  // Find the matches.
  int num = all_loc_preds.size();
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > match_indices;
    map<int, vector<float> > match_overlaps;

    // Find match between predictions and ground truth.
    // find用在map中
    // 当前图片的gt
    const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes.find(i)->second;
    {
      // Use prior bboxes to match against all ground truth.
      vector<int> temp_match_indices;
      vector<float> temp_match_overlaps;
      const int label = -1;

      //apply arm_loc_preds to prior_box
      // 最后一个label存放的是所有的结果
      const vector<NormalizedBBox>& arm_loc_preds = all_arm_loc_preds[i].find(label)->second;
      vector<NormalizedBBox> decode_prior_bboxes;
      bool clip_bbox = false;
      // 解码得到arm预测的boxes，decode_prior_bboxes
      DecodeBBoxes(prior_bboxes, prior_variances,
    		  code_type, encode_variance_in_target, clip_bbox,
			  arm_loc_preds, &decode_prior_bboxes);
      // 使用预测结果和gt进行匹配？我就是这么做的啊。
      // 得到匹配索引。匹配的refined prior，对gt进行编码。
      MatchBBox(gt_bboxes, decode_prior_bboxes, label, match_type, overlap_threshold,
                ignore_cross_boundary_bbox, &temp_match_indices,
                &temp_match_overlaps);
      if (share_location) {
      // map可以通过这种方式赋值，自动创建key = -1，这里只有一个key
        match_indices[label] = temp_match_indices;
        match_overlaps[label] = temp_match_overlaps;
      }
    }
    all_match_indices->push_back(match_indices);
    all_match_overlaps->push_back(match_overlaps);
  }
}