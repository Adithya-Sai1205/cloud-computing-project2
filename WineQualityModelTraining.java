import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityModelTraining {
    public static void main(String[] args) {
        // Initialize Spark
        SparkSession spark = SparkSession.builder().appName("WineQualityModelTraining").getOrCreate();
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        // Load training data
        String trainingDataPath = "s3://sqs.us-east-1.amazonaws.com/047412792150/MyQueue1/TrainingData.csv";
        Dataset<Row> trainingData = spark.read().option("header", "true").option("delimiter", ";").csv(trainingDataPath);

        // Prepare features and labels
        String[] featureColumns = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH",
                "sulphates", "alcohol"}; // Replace with actual feature names
        VectorAssembler assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(trainingData);
        Dataset<Row> labeledData = assembledData.select("features", "quality"); // Replace with actual label column name

        // Create and train the model
        LogisticRegression lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8);
        LogisticRegressionModel model = lr.fit(labeledData);

        // Save the model
        String modelPath = "s3://sqs.us-east-1.amazonaws.com/047412792150/MyQueue1/model";
        model.write().overwrite().save(modelPath);

        // Stop Spark
        spark.stop();
    }
}
