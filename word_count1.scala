package com.ex1

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.sql._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import org.apache.spark.sql.functions._


object word_count1 {

  final case class DF(WORD: String)

  def main(args: Array[String]) {

    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("word_count1")
      .master("local[7]")
      .config("spark.wordcount", "file:///C:/temp") 
      .getOrCreate()
      
    val startTimeMillis = System.currentTimeMillis()
     
    val input = spark.sparkContext.textFile("../Shakespeare.txt")
    val words = input.flatMap(x => x.split("\\W+"))
    val words1 = words.map(x => x.toLowerCase()).cache()
    val words2 = words1.map(x => DF(x.toString()))
    
    import spark.implicits._
    val df_words = words2.toDS()
    val topWords = df_words.groupBy("WORD").count().orderBy(desc("count")).cache()
    val count = df_words.count()
    val count_uni = topWords.select("count").map(x =>1)
    val count_unii = count_uni.count()
    
    topWords.show()
 
    print(s"total number of words: $count")
    print(s"\ntotal number of uni words: $count_unii")
    
    
    val top10 = topWords.take(10)
   
    println(top10(0))

    val endTimeMillis = System.currentTimeMillis()
    val durationSeconds = (endTimeMillis - startTimeMillis) 
    println(durationSeconds)
    
    spark.stop()
    

    
  }
  
}