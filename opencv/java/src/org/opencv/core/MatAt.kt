package org.opencv.core

import org.opencv.core.Mat.*
import java.lang.RuntimeException

// Safe conversion functions
private fun UByteArray.toByteArray(): ByteArray = ByteArray(this.size) { this[it].toByte() }
private fun UShortArray.toShortArray(): ShortArray = ShortArray(this.size) { this[it].toShort() }

// Extension functions with proper type handling
fun Mat.get(row: Int, col: Int, data: UByteArray): Int {
    val byteData = ByteArray(data.size)
    val result = this.get(row, col, byteData)
    for (i in data.indices) {
        data[i] = byteData[i].toUByte()
    }
    return result
}

fun Mat.get(indices: IntArray, data: UByteArray): Int {
    val byteData = ByteArray(data.size)
    val result = this.get(indices, byteData)
    for (i in data.indices) {
        data[i] = byteData[i].toUByte()
    }
    return result
}

fun Mat.put(row: Int, col: Int, data: UByteArray): Int {
    return this.put(row, col, data.toByteArray())
}

fun Mat.put(indices: IntArray, data: UByteArray): Int {
    return this.put(indices, data.toByteArray())
}

fun Mat.get(row: Int, col: Int, data: UShortArray): Int {
    val shortData = ShortArray(data.size)
    val result = this.get(row, col, shortData)
    for (i in data.indices) {
        data[i] = shortData[i].toUShort()
    }
    return result
}

fun Mat.get(indices: IntArray, data: UShortArray): Int {
    val shortData = ShortArray(data.size)
    val result = this.get(indices, shortData)
    for (i in data.indices) {
        data[i] = shortData[i].toUShort()
    }
    return result
}

fun Mat.put(row: Int, col: Int, data: UShortArray): Int {
    return this.put(row, col, data.toShortArray())
}

fun Mat.put(indices: IntArray, data: UShortArray): Int {
    return this.put(indices, data.toShortArray())
}

@Suppress("UNCHECKED_CAST")
inline fun <reified T> Mat.at(row: Int, col: Int): Atable<T> =
    when (T::class) {
        Byte::class, Double::class, Float::class, Int::class, Short::class -> this.at(
            T::class.java,
            row,
            col
        )
        UByte::class -> AtableUByte(this, row, col) as Atable<T>
        UShort::class -> AtableUShort(this, row, col) as Atable<T>
        else -> throw RuntimeException("Unsupported class type")
    }

@Suppress("UNCHECKED_CAST")
inline fun <reified T> Mat.at(idx: IntArray): Atable<T> =
    when (T::class) {
        Byte::class, Double::class, Float::class, Int::class, Short::class -> this.at(
            T::class.java,
            idx
        )
        UByte::class -> AtableUByte(this, idx) as Atable<T>
        UShort::class -> AtableUShort(this, idx) as Atable<T>
        else -> throw RuntimeException("Unsupported class type")
    }

class AtableUByte(val mat: Mat, val indices: IntArray) : Atable<UByte> {

    constructor(mat: Mat, row: Int, col: Int) : this(mat, intArrayOf(row, col))

    override fun getV(): UByte {
        val data = UByteArray(1)
        mat.get(indices, data)
        return data[0]
    }

    override fun setV(v: UByte) {
        val data = ubyteArrayOf(v)
        mat.put(indices, data)
    }

    override fun getV2c(): Tuple2<UByte> {
        val data = UByteArray(2)
        mat.get(indices, data)
        return Tuple2(data[0], data[1])
    }

    override fun setV2c(v: Tuple2<UByte>) {
        // Safe access to tuple values
        val val0 = v.get_0() ?: 0u.toUByte()
        val val1 = v.get_1() ?: 0u.toUByte()
        val data = ubyteArrayOf(val0, val1)
        mat.put(indices, data)
    }

    override fun getV3c(): Tuple3<UByte> {
        val data = UByteArray(3)
        mat.get(indices, data)
        return Tuple3(data[0], data[1], data[2])
    }

    override fun setV3c(v: Tuple3<UByte>) {
        val val0 = v.get_0() ?: 0u.toUByte()
        val val1 = v.get_1() ?: 0u.toUByte()
        val val2 = v.get_2() ?: 0u.toUByte()
        val data = ubyteArrayOf(val0, val1, val2)
        mat.put(indices, data)
    }

    override fun getV4c(): Tuple4<UByte> {
        val data = UByteArray(4)
        mat.get(indices, data)
        return Tuple4(data[0], data[1], data[2], data[3])
    }

    override fun setV4c(v: Tuple4<UByte>) {
        val val0 = v.get_0() ?: 0u.toUByte()
        val val1 = v.get_1() ?: 0u.toUByte()
        val val2 = v.get_2() ?: 0u.toUByte()
        val val3 = v.get_3() ?: 0u.toUByte()
        val data = ubyteArrayOf(val0, val1, val2, val3)
        mat.put(indices, data)
    }
}

class AtableUShort(val mat: Mat, val indices: IntArray) : Atable<UShort> {

    constructor(mat: Mat, row: Int, col: Int) : this(mat, intArrayOf(row, col))

    override fun getV(): UShort {
        val data = UShortArray(1)
        mat.get(indices, data)
        return data[0]
    }

    override fun setV(v: UShort) {
        val data = ushortArrayOf(v)
        mat.put(indices, data)
    }

    override fun getV2c(): Tuple2<UShort> {
        val data = UShortArray(2)
        mat.get(indices, data)
        return Tuple2(data[0], data[1])
    }

    override fun setV2c(v: Tuple2<UShort>) {
        val val0 = v.get_0() ?: 0u.toUShort()
        val val1 = v.get_1() ?: 0u.toUShort()
        val data = ushortArrayOf(val0, val1)
        mat.put(indices, data)
    }

    override fun getV3c(): Tuple3<UShort> {
        val data = UShortArray(3)
        mat.get(indices, data)
        return Tuple3(data[0], data[1], data[2])
    }

    override fun setV3c(v: Tuple3<UShort>) {
        val val0 = v.get_0() ?: 0u.toUShort()
        val val1 = v.get_1() ?: 0u.toUShort()
        val val2 = v.get_2() ?: 0u.toUShort()
        val data = ushortArrayOf(val0, val1, val2)
        mat.put(indices, data)
    }

    override fun getV4c(): Tuple4<UShort> {
        val data = UShortArray(4)
        mat.get(indices, data)
        return Tuple4(data[0], data[1], data[2], data[3])
    }

    override fun setV4c(v: Tuple4<UShort>) {
        val val0 = v.get_0() ?: 0u.toUShort()
        val val1 = v.get_1() ?: 0u.toUShort()
        val val2 = v.get_2() ?: 0u.toUShort()
        val val3 = v.get_3() ?: 0u.toUShort()
        val data = ushortArrayOf(val0, val1, val2, val3)
        mat.put(indices, data)
    }
}

// Component operators - using proper OpenCV Tuple API
operator fun <T> Tuple2<T>.component1(): T? = this.get_0()
operator fun <T> Tuple2<T>.component2(): T? = this.get_1()

operator fun <T> Tuple3<T>.component1(): T? = this.get_0()
operator fun <T> Tuple3<T>.component2(): T? = this.get_1()
operator fun <T> Tuple3<T>.component3(): T? = this.get_2()

operator fun <T> Tuple4<T>.component1(): T? = this.get_0()
operator fun <T> Tuple4<T>.component2(): T? = this.get_1()
operator fun <T> Tuple4<T>.component3(): T? = this.get_2()
operator fun <T> Tuple4<T>.component4(): T? = this.get_3()

// Helper functions
fun <T> T2(_0: T, _1: T): Tuple2<T> = Tuple2(_0, _1)
fun <T> T3(_0: T, _1: T, _2: T): Tuple3<T> = Tuple3(_0, _1, _2)
fun <T> T4(_0: T, _1: T, _2: T, _3: T): Tuple4<T> = Tuple4(_0, _1, _2, _3)
