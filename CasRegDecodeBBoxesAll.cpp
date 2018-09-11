void CasRegDecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, vector<LabelBBox>* all_decode_bboxes,
	const vector<LabelBBox>& all_arm_loc_preds) {
  CHECK_EQ(all_loc_preds.size(), num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  for (int i = 0; i < num; ++i) {
	//apply arm_loc_preds to prior_box
	const vector<NormalizedBBox>& arm_loc_preds = all_arm_loc_preds[i].find(-1)->second;
	vector<NormalizedBBox> decode_prior_bboxes;
	bool clip_bbox = false;
	DecodeBBoxes(prior_bboxes, prior_variances,
	  		code_type, variance_encoded_in_target, clip_bbox,
			arm_loc_preds, &decode_prior_bboxes);
    // Decode predictions into bboxes.
    LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
    for (int c = 0; c < num_loc_classes; ++c) {
//      int label = share_location ? -1 : c;
      int label = -1
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
      }
      const vector<NormalizedBBox>& label_loc_preds =
          all_loc_preds[i].find(label)->second;
      DecodeBBoxes(decode_prior_bboxes, prior_variances,
                   code_type, variance_encoded_in_target, clip,
                   label_loc_preds, &(decode_bboxes[label]));
    }
  }
}