package spark.linearregression

import org.apache.spark.ml.linalg.{DenseVector, Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    return v1.toArray.zip(v2.toArray).map( e => e._1 * e._2 ).reduce(_+_)
  }

  def dot(v: Vector, s: Double): Vector = {
    return Vectors.dense(v.toArray.map(_*s))
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    return Vectors.dense(v1.toArray.zip(v2.toArray).map( e => e._1 + e._2 ))
  }

  def fill(size: Int, fillVal: Double): Vector = {
    return Vectors.dense(Array.fill(size){fillVal})
  }
}