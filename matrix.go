/*
Package matrix implements a "mat" object, which behaves like a 2D array
or list in other programming languages. Under the hood, the mat object is a
flat slice, which provides for optimal performance in Go, while the methods
and constructors provide for a higher level of performance and abstraction
when compared to the "2D" slices of go (slices of slices). Due to it's internal
representation, row or column vectors can also be easily created by the Matf32
object, without a performance hit, by setting the number of rows or columns to
one.

All errors encountered in this package, such as attempting to access an
element out of bounds are treated as critical error, and thus, the code
immediately exits with signal 1. In such cases, the function/method in
which the error was encountered is printed to the screen, in addition
to the full stack trace, in order to help fix the issue rapidly.
*/
package matrix
