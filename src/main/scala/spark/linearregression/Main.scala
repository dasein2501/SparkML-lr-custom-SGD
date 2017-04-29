package spark.linearregression

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PolynomialExpansion, RegexTokenizer, VectorAssembler, VectorSlicer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {

    val spark = SparkSession.builder()
      .master("local")
      .appName("lab1")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame =  spark.sparkContext.textFile(filePath).toDF("rows")
    val Array(training, test) = obsDF.randomSplit(Array(0.7, 0.3), seed = 54321)

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("rows")
      .setOutputCol("tokens")
      .setPattern(",")

    val arr2Vect = new Array2Vector()
      .setInputCol("tokens")
      .setOutputCol("tokens2vector")

    val lSlicer = new VectorSlicer()
      .setInputCol("tokens2vector")
      .setOutputCol("label_vector")
      .setIndices(Array(0))

    val v2d = new Vector2DoubleUDF(v => v(0))
      .setInputCol("label_vector")
      .setOutputCol("label")


    val minLabel = 1922.0
    val lShifter = new DoubleUDF(label => label - minLabel)
      .setInputCol("label")
      .setOutputCol("label_shift")

    val fSlicer = new VectorSlicer()
      .setInputCol("tokens2vector")
      .setOutputCol("features")
      .setIndices(Array(1,3))

    val myLR = new MyLinearRegressionImpl().setFeaturesCol("features").setLabelCol("label_shift")

    val pipeline = new Pipeline()
      .setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))

    val model: PipelineModel = pipeline.fit(training)

    val modelSummary = model.stages(6).asInstanceOf[MyLinearModelImpl]

    // Run cross-validation, and choose the best set of parameters.
    val predictions = model.transform(test)

    //print rmse of our model
    println("RMSE: " + modelSummary.trainingError(modelSummary.trainingError.length-1))

    //do prediction - print first k
    predictions.select("features", "label_shift", "prediction").show(5)

  }
}