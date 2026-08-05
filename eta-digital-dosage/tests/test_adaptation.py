from eta_digital.adaptation.online_update import OnlineUpdater

def test_online_update_requires_valid_sample(predictor,training_frame):
    updater=OnlineUpdater(predictor); row=training_frame.iloc[0].copy(); row["sensor_quality"]=1.; assert updater.update(row)
    row["sensor_quality"]=0.1; assert not updater.update(row)
