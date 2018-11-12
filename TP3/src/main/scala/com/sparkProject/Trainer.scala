package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.SparkSession
import scala.math.pow


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    // Loading of the data set
    val df = spark.read.parquet("path_to_the_data_set")

    // Pipeline stages

    // Tokenize text column
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Remove stop words
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    // Compute TF
    val countVectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("tf")

    // Compute IDF
    val idf = new IDF()
      .setInputCol(countVectorizer.getOutputCol)
      .setOutputCol("tfidf")

    // Convert country labels in label indices
    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    // Convert currency labels in label indices
    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    // One-hot encoding of country indices
    val countryOneHotEncoder = new OneHotEncoder()
      .setInputCol(countryIndexer.getOutputCol)
      .setOutputCol("country_onehot")

    // One-hot encoding of currency indices
    val currencyOneHotEncoder = new OneHotEncoder()
      .setInputCol(currencyIndexer.getOutputCol)
      .setOutputCol("currency_onehot")

    // Combination of all features into a single vector
    val assembler = new VectorAssembler()
      .setInputCols(Array(idf.getOutputCol, "days_campaign", "hours_prepa", "goal",
                    countryOneHotEncoder.getOutputCol, currencyOneHotEncoder.getOutputCol))
      .setOutputCol("features")

    // Logistic regression estimator
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    // Creation of the pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf, countryIndexer, currencyIndexer, countryOneHotEncoder, currencyOneHotEncoder, assembler, lr))

    // Split of the dataset into a training set and a test set
    val splits = df.randomSplit(Array(0.9, 0.1), seed=1)

    val (training, test) = (splits(0), splits(1))

    // Multi-class classification evaluator with a f1-score metric
    val multiClassEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // Parameter grid to perform cross-validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(pow(10, -8), pow(10, -6), pow(10, -4), pow(10, -2)))
      .addGrid(countVectorizer.minDF, Array(55.toDouble, 75.toDouble, 95.toDouble))
      .build()

    // Cross-validation
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(multiClassEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // Model training
    val model = trainValidationSplit.fit(training)

    // Compute predictions
    val df_WithPredictions = model.transform(test)

    // Display labels and predictions
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    // Display f1-score of the model selected by cross-validation
    println("f1 score : " + multiClassEvaluator.evaluate(df_WithPredictions))

    // Save the trained model
    model.write.overwrite().save("path_to_the_saved_model")
  }
}
