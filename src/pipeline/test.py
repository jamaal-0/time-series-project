from .xx import ForecastPipeline


pipeline = ForecastPipeline("/Users/v/Data Science Projects/time-series-project/data/raw.csv")

clean_data = pipeline.clean()

weekly = pipeline.aggregate_weekly()

featured = pipeline.create_features(weekly)

encoded = pipeline.encode_categorical(featured)

model, mae = pipeline.train_model(encoded)

print("MAE:", mae)