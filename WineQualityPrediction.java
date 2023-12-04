import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPrediction {
    public static void main(String[] args) {
        // Initialize Spark
        SparkSession spark = SparkSession.builder().appName("WineQualityPrediction").getOrCreate();

        // Load the saved model
        String modelPath = "s3://sqs.us-east-1.amazonaws.com/047412792150/MyQueue1/model";
        PipelineModel loadedModel = PipelineModel.load(modelPath);

        // Load validation data
        String validationDataPath = "s3://sqs.us-east-1.amazonaws.com/047412792150/MyQueue1/ValidationData.csv";
        Dataset<Row> validationData = spark.read().option("header", "true").option("delimiter", ";").csv(validationDataPath);

        // Perform predictions
        Dataset<Row> predictions = loadedModel.transform(validationData);

        // Show predictions or save them to a file
        predictions.show();

        // Stop Spark
        spark.stop();
    }
}
