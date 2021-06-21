package com.ex1

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.sql._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator



object clustering {
  
  case class DataSet(X:Double, Y:Double)
  
    def mapper(line:String): DataSet = {
    val fields = line.split("\\s+")
    val data:DataSet = DataSet(fields(0).toDouble, fields(1).toDouble)
    return data
    }
  
  
   def main(args: Array[String]) {
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    val spark = SparkSession
      .builder
      .appName("clusteringL")
      .master("local[4]")
      .config("spark.clustering", "file:///C:/temp") 
      .getOrCreate()

    import spark.implicits._
    val lines = spark.sparkContext.textFile("../C3.txt")
    val dataset = lines.map(mapper).toDS()
    
    dataset.show()
    
    val assembler = new VectorAssembler()
    .setInputCols(Array("X","Y"))
    .setOutputCol("features")
    val output = assembler.transform(dataset)
    output.show(5)
    
    val startTimeMillis = System.currentTimeMillis()
  
//    val kmeans = new KMeans()
//    .setK(3)
//    .setFeaturesCol("features")
//    .setInitMode("k-means||")
//    val model = kmeans.fit(output)  
    
    val bkm = new BisectingKMeans()
    .setK(3)
    .setFeaturesCol("features")
    val model = bkm.fit(output)

   
    val endTimeMillis = System.currentTimeMillis()
    val durationSeconds = (endTimeMillis - startTimeMillis) 
    println(durationSeconds)
    
  
    val predictions = model.transform(output)
    val evaluator = new ClusteringEvaluator()
    val sed = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $sed")
    
    val ss= predictions

    model.clusterCenters.foreach(println)
    
    spark.stop()
    
  }
  
  
}