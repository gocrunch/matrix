package matrix

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"reflect"
	"strconv"

	"github.com/gorgonia/vecf64"
)

/*
Matf64 is the main struct of this library. Matf64 is a essentially a 1D slice
(a []float64) that contains two integers, representing rows and columns,
which allow it to behave as if it was a 2D slice. This allows for higher
performance and flexibility for the users of this library, at the expense
of some bookkeeping that is done here.

The fields of this struct are not directly accessible, and they may only
change by the use of the various methods in this library.
*/
type Matf64 struct {
	r, c int
	vals []float64
}

/*
Newf64 is the primary constructor for the "Matf64" object. New is a variadic function,
expecting 0 to 2 integers, with differing behavior as follows:

	m := matrix.Newf64()

m is now an empty &Matf64{}, where the number of rows,
columns and the length and capacity of the underlying
slice are all zero. This is mostly for internal use.

	m := matrix.Newf64(x)

m is a x by x (square) matrix, with the underlying
slice of length x, and capacity 2x.

	m := matrix.Newf64(x, y)

m is an x by y matrix, with the underlying slice of
length xy, and capacity of 2xy.
*/
func Newf64(dims ...int) *Matf64 {
	m := &Matf64{}
	switch len(dims) {
	case 0:
		m = &Matf64{
			0,
			0,
			make([]float64, 0),
		}
	case 1:
		m = &Matf64{
			dims[0],
			dims[0],
			make([]float64, dims[0]*dims[0], 2*dims[0]*dims[0]),
		}
	case 2:
		m = &Matf64{
			dims[0],
			dims[1],
			make([]float64, dims[0]*dims[1], 2*dims[0]*dims[1]),
		}
	default:
		s := "\nIn matrix.%s, expected 0 to 2 arguments, but received %d arguments."
		s = fmt.Sprintf(s, "Newf64()", len(dims))
		printErr(s)
	}
	return m
}

/*
Eyef64 returns the identity matrix
*/
func Eyef64(x int) *Matf64 {
	m := Newf64(x)
	for i := 1; i < x; i++ {
		m.vals[i*i-1] = 1.0
	}
	return m
}

/*
Matf64FromData creates a mat object from a []float64 or a [][]float64 slice.
This function is designed to do the "right thing" based on the type of
the slice passed to it. The "right thing" based on each possible case
is as follows:

Assume that s is a [][]float64, and v is a []float64 for the examples
below.

	x := matrix.Matf64FromData(v)

In this case, x.Dims() is (1, len(v)), and the values in x are the same
as the values in v. x is essentially a row vector.

Alternatively, this function can be invoked as:

	x := matrix.Matf64FromData(v, a)

In this case, x.Dims() is (a, 1), and the values in x are the same
as the values in v. x is essentially a column vector. Note that a
must be equal to len(v).

Finally for the case where the data is a []float64, the function can be
invoked as:

	x := matrix.Matf64FromData(v, a, b)

In this case, x.Dims() is (a, b), and the values in x are the same as
the values in v. Note that a*b must be equal to len(v). Also note that
this is equivalent to:

    x := matrix.Matf64FromData(v).reshape(a,b)

This function can also be invoked with data that is stored in a 2D
slice ([][]float64). Just as the []float64 case, there are three
possibilities:

	x := matrix.Matf64FromData(s)

In this case, x.Dims() is (len(s), len(s[0])), and the values in x
are the same as the values in s. It is assumed that s is not jagged.

Another form to call this function with a 2D slice of data is:

	x := matrix.Matf64FromData(s, a)

In this case, x.Dims() is (a, a), and the values in x are the same
as the values in s. Note that the total number of elements in s
must be exactly a*a.

Finally, this function can be called as:

	x := matrix.Matf64FromData(s, a, b)

In this case, x.Dims() is (a, b), and the values in x are the same
as the values in s. Note that the total number of elements in s
must be exactly a*b. Also note that this is equivalent to:

	x := matrix.Matf64FromData(s).Reshape(a, b)

Choose the format that suits your needs, as there is no performance
difference between the two forms.
*/
func Matf64FromData(oneOrTwoDSlice interface{}, dims ...int) *Matf64 {
	switch v := oneOrTwoDSlice.(type) {
	case []float64:
		return matf64FromOneDSliceHelper(v, dims)
	case [][]float64:
		return matf64FromTwoDSliceHelper(v, dims)
	default:
		s := "\nIn matrix.%s, expected input data of type []float64 or\n"
		s += "[][]float64, However, data of type \"%v\" was received."
		s = fmt.Sprintf(s, "Matf64FromData()", reflect.TypeOf(v))
		printErr(s)
	}
	return nil
}

func matf64FromOneDSliceHelper(v []float64, dims []int) *Matf64 {
	m := Newf64()
	switch len(dims) {
	case 0:
		m.vals = make([]float64, len(v), len(v)*2)
		copy(m.vals, v)
		m.r, m.c = 1, len(v)
	case 1:
		if dims[0] != len(v) {
			s := "\nIn matrix.%s, a 1D slice of data and a single int were passed.\n"
			s += "However the int (%d) is not equal to the length of the data (%d)."
			s = fmt.Sprintf(s, "Matf64FromData()", dims[0], len(v))
			printHelperErr(s)
		}
		m.vals = make([]float64, dims[0], dims[0]*2)
		copy(m.vals, v)
		m.r, m.c = dims[0], 1
	case 2:
		if dims[0]*dims[1] != len(v) {
			s := "\nIn matrix.%s, a 1D slice of data and two ints were passed.\n"
			s += "However, the product of the two ints (%d, %d) does not equal\n"
			s += "the number of elements in the data slice, %d. They must be equal."
			s = fmt.Sprintf(s, "Matf64FromData()", dims[0]*dims[1], len(v))
			printHelperErr(s)
		}
		m.vals = make([]float64, dims[0]*dims[1], dims[0]*dims[1]*2)
		copy(m.vals, v)
		m.r, m.c = dims[0], dims[1]
	default:
		s := "\nIn matrix.%s, a 1D slice of data and %d ints were passed.\n"
		s += "This function expects 0 to 2 integers. Please review the docs for\n"
		s += "this function and adjust the number of integers based on the\n"
		s += "desired output."
		s = fmt.Sprintf(s, "Matf64FromData()", len(dims))
		printHelperErr(s)
	}
	return m
}

func matf64FromTwoDSliceHelper(v [][]float64, dims []int) *Matf64 {
	m := Newf64()
	switch len(dims) {
	case 0:
		m.vals = make([]float64, len(v)*len(v[0]), len(v)*len(v[0])*2)
		for i := range v {
			for j := range v[i] {
				m.vals[i*len(v[0])+j] = v[i][j]
			}
		}
		m.r, m.c = len(v), len(v[0])
	case 1:
		if dims[0]*dims[0] != len(v)*len(v[0]) {
			s := "\nIn matrix.%s, a 2D slice of data and 1 int were passed.\n"
			s += "This would generate a %d by %d Matf64. However, %d*%d does not\n"
			s += "equal the number of elements in the passed 2D slice, %d.\n"
			s += "Note that this function expects a non-jagged 2D slice, and\n"
			s += "is assumed that every row in the passed 2D slice contains\n"
			s += "%d elements."
			s = fmt.Sprintf(s, "Matf64FromData()", dims[0], dims[0], dims[0], dims[0],
				len(v)*len(v[0]), len(v[0]))
			printHelperErr(s)
		}
		m.vals = make([]float64, dims[0]*dims[0], dims[0]*dims[0]*2)
		for i := range v {
			for j := range v[i] {
				m.vals[i*len(v[0])+j] = v[i][j]
			}
		}
		m.r, m.c = dims[0], dims[0]
	case 2:
		if dims[0] != len(v) || dims[1] != len(v[0]) {
			s := "\nIn matrix.%s, a 2D slice of data and 2 ints were passed.\n"
			s += "However, the requested number of rows and columns (%d and %d)\n"
			s += "of the resultant Matf64 does not match the length and width of\n"
			s += "the data slice (%d and %d)."
			s = fmt.Sprintf(s, "Matf64FromData()", dims[0], dims[1], len(v), len(v[0]))
			printHelperErr(s)
		}
		m.vals = make([]float64, dims[0]*dims[1], dims[0]*dims[1]*2)
		for i := range v {
			for j := range v[i] {
				m.vals[i*len(v[0])+j] = v[i][j]
			}
		}
		m.r, m.c = len(v), len(v[0])
	default:
		s := "\nIn matrix.%s, a 2D slice of data and %d ints were passed.\n"
		s += "However, this function expects 0 to 2 ints. Review the docs for\n"
		s += "this function and adjust the number of integers passed accordingly."
		s = fmt.Sprintf(s, "Matf64FromData()", len(dims))
		printHelperErr(s)
	} // switch len(dims) for case [][]float64
	return m
}

/*
Matf64FromCSV creates a mat object from a CSV (comma separated values) file. Here, we
assume that the number of rows of the resultant mat object is equal to the
number of lines, and the number of columns is equal to the number of entries
in each line. As before, we make sure that each line contains the same number
of elements.

The file to be read is assumed to be very large, and hence it is read one line
at a time. This results in some major inefficiencies, and it is recommended
that this function be used sparingly, and not as a major component of your
library/executable.

Unlike other mat creation functions in this package, the capacity of the mat
object created here is the same as its length since we assume the mat to
be very large.
*/
func Matf64FromCSV(filename string) *Matf64 {
	f, err := os.Open(filename)
	if err != nil {
		s := "\nIn matrix.%s, cannot open %s due to error: %v.\n"
		s = fmt.Sprintf(s, "Matf64FromCSV()", filename, err)
		printErr(s)
	}
	defer f.Close()
	r := csv.NewReader(f)
	// I am going with the assumption that a mat loaded from a CSV is going to
	// be large. So, we are going to read one line, and determine the number
	// of columns based on the number of comma separated entries in that line.
	// Then we will read the rest of the lines one at a time, checking that the
	// number of entries in each line is the same as the first line.
	str, err := r.Read()
	if err != nil {
		s := "\nIn matrix.%s, cannot read from %s due to error: %v.\n"
		s = fmt.Sprintf(s, "Matf64FromCSV()", filename, err)
		printErr(s)
	}
	// Start with one row, and set the number of entries per row
	m := Newf64()
	m.r, m.c = 1, len(str)
	row := make([]float64, len(str))
	for {
		for i := range str {
			row[i], err = strconv.ParseFloat(str[i], 64)
			if err != nil {
				s := "\nIn matrix.%s, item %d in line %d is %s, which cannot\n"
				s += "be converted to a float64 due to: %v"
				s = fmt.Sprintf(s, "Matf64FromCSV()", i, m.r, str[i], err)
				printErr(s)
			}
		}
		m.vals = append(m.vals, row...)
		// Read the next line. If there is one, increment the number of rows
		str, err = r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			s := "\nIn matrix.%s, cannot read from %s due to error: %v.\n"
			s = fmt.Sprintf(s, "Matf64FromCSV()", filename, err)
			printErr(s)
		}
		m.r++
	}
	return m
}

/*
RandMatf64 returns a Matf64 whose elements have random values. There are 3 ways to call
RandMatf64:

	m := matrix.RandMatf64(2, 3)

With this call, m is a 2X3 Matf64 whose elements have values randomly selected from
the range (0, 1], (includes 0, but excludes 1).

	m := matrix.RandMatf64(2, 3, x)

With this call, m is a 2X3 Matf64 whose elements have values randomly selected from
the range (0, x], (includes 0, but excludes x).

	m := matrix.RandMatf64(2, 3, x, y)

With this call, m is a 2X3 Matf64 whose elements have values randomly selected from
the range (x, y], (includes x, but excludes y). In this case, x must be strictly
less than y.
*/
func RandMatf64(r, c int, args ...float64) *Matf64 {
	m := Newf64(r, c)
	switch len(args) {
	case 0:
		for i := 0; i < m.r*m.c; i++ {
			m.vals[i] = rand.Float64()
		}
	case 1:
		to := args[0]
		for i := 0; i < m.r*m.c; i++ {
			m.vals[i] = rand.Float64() * to
		}
	case 2:
		from := args[0]
		to := args[1]
		if !(from < to) {
			s := "\nIn matrix.%s the first argument, %f, is not less than the\n"
			s += "second argument, %f. The first argument must be strictly\n"
			s += "less than the second.\n"
			s = fmt.Sprintf(s, "RandMatf64()", from, to)
			printErr(s)
		}
		for i := 0; i < m.r*m.c; i++ {
			m.vals[i] = rand.Float64()*(to-from) + from
		}
	default:
		s := "\nIn matrix.%s expected 0 to 2 arguments, but received %d."
		s = fmt.Sprintf(s, "RandMatf64()", len(args))
		printErr(s)
	}
	return m
}

/*
Reshape changes the row and the columns of the mat object as long as the total
number of values contained in the mat object remains constant. The order and
the values of the mat does not change with this function.
*/
func (m *Matf64) Reshape(rows, cols int) *Matf64 {
	if rows*cols != m.r*m.c {
		s := "\nIn %s, The total number of entries of the old and new shape\n"
		s += "must match. The Old Matf64 had a shape of row = %d, col = %d,\n"
		s += "which is not equal to the requested shape of row, col = %d, %d\n"
		s = fmt.Sprintf(s, "Reshape()", m.r, m.c, rows, cols)
		printErr(s)
	} else {
		m.r = rows
		m.c = cols
	}
	return m
}

/*
Shape returns the number of rows and columns of a mat object.
*/
func (m *Matf64) Shape() (int, int) {
	return m.r, m.c
}

/*
ToSlice1D returns the values contained in a mat object as a 1D slice of float64s.
*/
func (m *Matf64) ToSlice1D() []float64 {
	s := make([]float64, len(m.vals))
	copy(s, m.vals)
	return s
}

/*
ToSlice2D returns the values of a mat object as a 2D slice of float64s.
*/
func (m *Matf64) ToSlice2D() [][]float64 {
	s := make([][]float64, m.r)
	for i := range s {
		s[i] = make([]float64, m.c)
		for j := range s[i] {
			s[i][j] = m.vals[i*m.c+j]
		}
	}
	return s
}

/*
ToCSV creates a file with the passed name, and writes the content of a mat
object to it, by putting each row in a single comma separated line. The
number of entries in each line is equal to the columns of the mat object.
*/
func (m *Matf64) ToCSV(fileName string) {
	f, err := os.Create(fileName)
	if err != nil {
		s := "\nIn %s, cannot open %s due to error: %v.\n"
		s = fmt.Sprintf(s, "ToCSV()", fileName, err)
		printErr(s)
	}
	defer f.Close()
	str := ""
	idx := 0
	for i := 0; i < m.r; i++ {
		for j := 0; j < m.c; j++ {
			str += strconv.FormatFloat(m.vals[idx], 'e', 14, 64)
			if j+1 != m.c {
				str += ","
			}
			idx++
		}
		if i+1 != m.r {
			str += "\n"
		}
	}
	_, err = f.Write([]byte(str))
	if err != nil {
		s := "\nIn %s, cannot write to %s due to error: %v.\n"
		s = fmt.Sprintf(s, "ToCSV()", fileName, err)
		printErr(s)
	}
}

/*
Get returns a pointer to the float64 stored in the given row and column.
*/
func (m *Matf64) Get(r, c int) float64 {
	return m.vals[r*m.c+c]
}

/*
Set sets the value of a mat at a given row and column to a given
value.
*/
func (m *Matf64) Set(r, c int, val float64) *Matf64 {
	m.vals[r*m.c+c] = val
	return m
}

/*
SetAll sets all values of a mat to the passed float64 value.
*/
func (m *Matf64) SetAll(val float64) *Matf64 {
	for i := range m.vals {
		m.vals[i] = val
	}
	return m
}

/*
Map applies a given function to each element of a mat object. The given
function must take a pointer to a float64, and return nothing. For eaxmple,
lets say that we wish to take the error function of each element of a Matf64. The
following would do this:

	m.Map(func(i *float64) {
		*i = math.Erf(*i)
	})
*/
func (m *Matf64) Map(f func(*float64)) *Matf64 {
	for i := range m.vals {
		f(&m.vals[i])
	}
	return m
}

/*
SetCol Sets all elements in a given column to the passed value(s). Negative
index values are allowed. For  example:

	m.SetCol(-1, 2.0)

sets all values of m's last column to 2.0. It is also possible to pass a slice
of float64 to this function, all the elements of the chosen column will be
set to the corresponding values in the slice. For example:

	m := Newf64(2, 2).SetCol(0, []float64{1.0, 2.0})

sets to values in the first column of m to 1.0 and 2.0 respectively. Note that
in this case, the length of the passed slice must match exactly the number of
elements in m's column, i.e. the number of rows of m.
*/
func (m *Matf64) SetCol(col int, floatOrSlice interface{}) *Matf64 {
	switch val := floatOrSlice.(type) {
	case float64:
		if (col >= m.c) || (col < -m.c) {
			s := "\nIn %s the requested column %d is outside of bounds [%d, %d)\n"
			s = fmt.Sprintf(s, "SetCol()", col, m.c, m.c)
			printErr(s)
		}
		if col >= 0 {
			for r := 0; r < m.r; r++ {
				m.vals[r*m.c+col] = val
			}
		} else {
			for r := 0; r < m.r; r++ {
				m.vals[r*m.c+(m.c+col)] = val
			}
		}
	case []float64:
		if len(val) != m.r {
			s := "\nIn %s the length of the passed slice is %d, which does\n"
			s += "not match the number of rows in the receiver, %d."
			s = fmt.Sprintf(s, "SetCol()", len(val), m.r)
			printErr(s)
		}
		if col >= 0 {
			for r := 0; r < m.r; r++ {
				m.vals[r*m.c+col] = val[r]
			}
		} else {
			for r := 0; r < m.r; r++ {
				m.vals[r*m.c+(m.c+col)] = val[r]
			}
		}
	default:
		s := "\nIn %s, the passed value must be a float64 or []float64.\n"
		s += "However, value of type  %v was received.\n"
		s = fmt.Sprintf(s, "SetCol()", reflect.TypeOf(val))
		printErr(s)
	}
	return m
}

/*
SetRow Sets all elements in a given column to the passed value(s). Negative
index values are allowed. For  example:

	m.SetRow(-1, 2.0)

sets all values of m's last row to 2.0. It is also possible to pass a slice
of float64 to this function, all the elements of the chosen row will be
set to the corresponding values in the slice. For example:

	m := Newf64(2, 2).SetRow(0, []float64{1.0, 2.0})

sets to values in the first row of m to 1.0 and 2.0 respectively. Note that
in this case, the length of the passed slice must match exactly the number of
elements in m's row, i.e. the number of cols of m.
*/
func (m *Matf64) SetRow(row int, floatOrSlice interface{}) *Matf64 {
	switch val := floatOrSlice.(type) {
	case float64:
		if (row >= m.r) || (row < -m.r) {
			s := "\nIn %s, row %d is outside of the bounds [-%d, %d)\n"
			s = fmt.Sprintf(s, "SetRow()", row, m.r, m.r)
			printErr(s)
		}
		if row >= 0 {
			for r := 0; r < m.c; r++ {
				m.vals[row*m.c+r] = val
			}
		} else {
			for r := 0; r < m.c; r++ {
				m.vals[(m.r+row)*m.c+r] = val
			}
		}
	case []float64:
		if len(val) != m.c {
			s := "\nIn %s the length of the passed slice is %d, which does\n"
			s += "not match the number of columns in the receiver, %d."
			s = fmt.Sprintf(s, "SetRow()", len(val), m.c)
			printErr(s)
		}
		if row >= 0 {
			for r := 0; r < m.c; r++ {
				m.vals[row*m.c+r] = val[r]
			}
		} else {
			for r := 0; r < m.c; r++ {
				m.vals[(m.r+row)*m.c+r] = val[r]
			}
		}
	default:
		s := "\nIn %s, the passed value must be a float64 or []float64.\n"
		s += "However, value of type  %v was received.\n"
		s = fmt.Sprintf(s, "SetRow()", reflect.TypeOf(val))
		printErr(s)
	}
	return m
}

/*
Col returns a new mat object whose values are equal to a column of the original
mat object. The number of Rows of the returned mat object is equal to the
number of rows of the original mat, and the number of columns is equal to 1.

This function supports negative indexing. For example,

	v := m.Col(-1)

returns the last column of m.
*/
func (m *Matf64) Col(x int) *Matf64 {
	if (x >= m.c) || (x < -m.c) {
		s := "\nIn %s the requested column %d is outside of bounds [-%d, %d)\n"
		s = fmt.Sprintf(s, "Col()", x, m.c, m.c)
		printErr(s)
	}
	v := Newf64(m.r, 1)
	if x >= 0 {
		for r := 0; r < m.r; r++ {
			v.vals[r] = m.vals[r*m.c+x]
		}
	} else {
		for r := 0; r < m.r; r++ {
			v.vals[r] = m.vals[r*m.c+(m.c+x)]
		}
	}
	return v
}

/*
Row returns a new mat object whose values are equal to a row of the original
mat object. The number of Rows of the returned mat object is equal to 1, and
the number of columns is equal to the number of columns of the original mat.

This function supports negative indexing. For example,

	v := m.Row(-1)

returns the last row of m.
*/
func (m *Matf64) Row(x int) *Matf64 {
	if (x >= m.r) || (x < -m.r) {
		s := "\nIn %s, row %d is outside of the bounds [-%d, %d)\n"
		s = fmt.Sprintf(s, "Row()", x, m.r, m.r)
		printErr(s)
	}
	v := Newf64(1, m.c)
	if x >= 0 {
		for r := 0; r < m.c; r++ {
			v.vals[r] = m.vals[x*m.c+r]
		}
	} else {
		for r := 0; r < m.c; r++ {
			v.vals[r] = m.vals[(m.r+x)*m.c+r]
		}
	}
	return v
}

/*
Min returns the index and the value of the smallest float64 in a Matf64. This
method can be called in one of two ways:

	idx, val := m.Min()

will return the index, and value of the smallest float64 in m. We can also
specify the exact row and column for which we want the minimum index and
values:

	idx, val := m.Min(0, 3) // Get the min index and value of the 4th row
	idx, val := m.Min(1, 2) // Get the min index and value of the 3rd column

Note that negative index values are not supported at this time. Also note that
in the case where multiple values are the maximum, the index of the first
encountered value is returned.
*/
func (m *Matf64) Min(args ...int) (index int, minVal float64) {
	switch len(args) {
	case 0:
		index = 0
		minVal = m.vals[0]
		for i := 1; i < len(m.vals); i++ {
			if m.vals[i] < minVal {
				minVal = m.vals[i]
				index = i
			}
		}
	case 2:
		axis, slice := args[0], args[1]
		switch axis {
		case 0:
			if (slice >= m.r) || (slice < 0) {
				s := "\nIn %s the row %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Min()", slice, m.r)
				printErr(s)
			}
			index = 0
			minVal = m.vals[slice*m.c]
			for i := 1; i < m.c; i++ {
				if m.vals[slice*m.c+i] < minVal {
					minVal = m.vals[slice*m.c+i]
					index = i
				}
			}
		case 1:
			if (slice >= m.c) || (slice < 0) {
				s := "\nIn %s the column %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Min()", slice, m.c)
				printErr(s)
			}
			index = 0
			minVal = m.vals[slice]
			for i := 1; i < m.r; i++ {
				if m.vals[i*m.c+slice] < minVal {
					minVal = m.vals[i*m.c+slice]
					index = i
				}
			}
		default:
			s := "\nIn %s, the first argument must be 0 or 1, however %d "
			s += "was received.\n"
			s = fmt.Sprintf(s, "Min()", axis)
			printErr(s)
		} // Switch on axis
	default:
		s := "\nIn %s, 0 or 2 arguments expected, but %d was received.\n"
		s = fmt.Sprintf(s, "Min()", len(args))
		printErr(s)
	} // switch on len(args)
	return index, minVal
}

/*
Max returns the index and the value of the biggest float64 in a Matf64. This
method can be called in one of two ways:

	idx, val := m.Max()

will return the index, and value of the biggest float64 in m. We can also
specify the exact row and column for which we want the minimum index and
values:

	idx, val := m.Max(0, 3) // Get the max index and value of the 4th row
	idx, val := m.Max(1, 2) // Get the max index and value of the 3rd column

Note that negative index values are not supported at this time. Also note that
in the case where multiple values are the maximum, the index of the first
encountered value is returned.
*/
func (m *Matf64) Max(args ...int) (index int, maxVal float64) {
	switch len(args) {
	case 0:
		index = 0
		maxVal = m.vals[0]
		for i := 1; i < len(m.vals); i++ {
			if m.vals[i] > maxVal {
				maxVal = m.vals[i]
				index = i
			}
		}
	case 2:
		axis, slice := args[0], args[1]
		switch axis {
		case 0:
			if (slice >= m.r) || (slice < 0) {
				s := "\nIn %s the row %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Max()", slice, m.r)
				printErr(s)
			}
			index = 0
			maxVal = m.vals[slice*m.c]
			for i := 1; i < m.c; i++ {
				if m.vals[slice*m.c+i] > maxVal {
					maxVal = m.vals[slice*m.c+i]
					index = i
				}
			}
		case 1:
			if (slice >= m.c) || (slice < 0) {
				s := "\nIn %s the column %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Max()", slice, m.c)
				printErr(s)
			}
			index = 0
			maxVal = m.vals[slice]
			for i := 1; i < m.r; i++ {
				if m.vals[i*m.c+slice] > maxVal {
					maxVal = m.vals[i*m.c+slice]
					index = i
				}
			}
		default:
			s := "\nIn %s, the first argument must be 0 or 1, however %d "
			s += "was received.\n"
			s = fmt.Sprintf(s, "Max()", axis)
			printErr(s)
		} // Switch on axis
	default:
		s := "\nIn %s, 0 or 2 arguments expected, but %d was received.\n"
		s = fmt.Sprintf(s, "Max()", len(args))
		printErr(s)
	} // switch on len(args)
	return index, maxVal
}

/*
Equals checks to see if two mat objects are equal. That mean that the two mats
have the same number of rows, same number of columns, and have the same float64
in each entry at a given index.
*/
func (m *Matf64) Equals(n *Matf64) bool {
	if m.r != n.r {
		return false
	}
	if m.c != n.c {
		return false
	}
	for i := 0; i < m.r*m.c; i++ {
		if m.vals[i] != n.vals[i] {
			return false
		}
	}
	return true
}

/*
Copy returns a duplicate of a mat object. The returned copy is "deep", meaning
that the object can be manipulated without effecting the original mat object.
*/
func (m *Matf64) Copy() *Matf64 {
	n := Newf64(m.r, m.c)
	copy(n.vals, m.vals)
	return n
}

/*
T returns the transpose of the original matrix. The transpose of a mat object
is defined in the usual manner, where every value at row x, and column y is
placed at row y, and column x. The number of rows and column of the transposed
mat are equal to the number of columns and rows of the original matrix,
respectively. This method creates a new mat object, and the original is
left intact.
*/
func (m *Matf64) T() *Matf64 {
	if m.isRowVector() || m.isColVector() {
		m.r, m.c = m.c, m.r
		return m
	}
	n := f64Pool.get()
	defer f64Pool.put(n)

	if len(n.vals) < m.c*m.r {
		n.vals = make([]float64, m.c*m.r)
	}
	idx := 0
	for i := 0; i < m.c; i++ {
		for j := 0; j < m.r; j++ {
			n.vals[idx] = m.vals[j*m.c+i]
			idx++
		}
	}
	m.r, m.c = m.c, m.r
	copy(m.vals, n.vals)
	return m
}

func (m *Matf64) isRowVector() bool {
	if m.r == 1 {
		return true
	}
	return false
}

func (m *Matf64) isColVector() bool {
	if m.c == 1 {
		return true
	}
	return false
}

/*
All checks if a supplied function is true for all elements of a mat object.
For instance, consider

	m.All(matrix.Positivef64)

will return true if and only if all elements in m are positive.
*/
func (m *Matf64) All(f func(*float64) bool) bool {
	for i := range m.vals {
		if !f(&m.vals[i]) {
			return false
		}
	}
	return true
}

/*
Any checks if a supplied function is true for one elements of a mat object.
For instance,

	m.Any(matrix.Positivef64)

would be true if at least one element of the mat object is positive.
*/
func (m *Matf64) Any(f func(*float64) bool) bool {
	for i := range m.vals {
		if f(&m.vals[i]) {
			return true
		}
	}
	return false
}

/*
Mul carries the multiplication operation between each element of the receiver
and an object passed to it. Based on the type of the passed object, the results
of this method changes:

If the passed object is a float64, then each element is multiplied by it:

	m := matrix.Newf64(2, 3).SetAll(5.0)
	m.Mul(2.0)

This will result in all values of m being 10.0.
The passed Object can also be a Matf64, in which case each element of the receiver
are multiplied by the corresponding element of the passed Matf64. Note that the
passed Matf64 must have the same shape as the receiver.

	m := matrix.Newf64(2, 3).SetAll(10.0)
	n := m.Copy()
	m.Mul(n)

This will result in each element of m being 100.0.

Note: For the matrix cross product see the Dot() method.
*/
func (m *Matf64) Mul(float64OrMatf64 interface{}) *Matf64 {
	switch v := float64OrMatf64.(type) {
	case float64:
		for i := range m.vals {
			m.vals[i] *= v
		}
	case *Matf64:
		if v.r != m.r {
			s := "\nIn %s, the number of the rows of the receiver is %d\n"
			s += "but the number of rows of the passed mat is %d. They must\n"
			s += "match.\n"
			s = fmt.Sprintf(s, "Mul()", m.r, v.r)
			printErr(s)
		}
		if v.c != m.c {
			s := "\nIn %s, the number of the columns of the receiver is %d\n"
			s += "but the number of columns of the passed mat is %d. They must\n"
			s += "match.\n"
			s = fmt.Sprintf(s, "Mul()", m.c, v.c)
			printErr(s)
		}
		vecf64.Mul(m.vals, v.vals)
	default:
		s := "\nIn %s, the passed value must be a float64 or *Matf64.\n"
		s += "However, value of type  \"%v\" was received.\n"
		s = fmt.Sprintf(s, "Mul()", reflect.TypeOf(v))
		printErr(s)
	}
	return m
}

/*
Add carries the addition operation between each element of the receiver
and an object passed to it. Based on the type of the passed object, the results
of this method changes:

If the passed object is a float64, then it is added to each element:

	m := matrix.Newf64(2, 3).SetAll(5.0)
	m.Add(2.0)

This will result in all values of m being 7.0.
The passed Object can also be a Matf64, in which case each element of the element
of the passed Matf64 is added to the corresponding element of the receiver. Note
that the passed Matf64 must have the same shape as the receiver.

	m := matrix.Newf64(2, 3).SetAll(10.0)
	n := m.Copy()
	m.Add(n)

This will result in each element of m being 20.0.
*/
func (m *Matf64) Add(float64OrMatf64 interface{}) *Matf64 {
	switch v := float64OrMatf64.(type) {
	case float64:
		for i := range m.vals {
			m.vals[i] += v
		}
	case *Matf64:
		if v.r != m.r {
			s := "\nIn %s, the number of the rows of the receiver is %d\n"
			s += "but the number of rows of the passed mat is %d. They must\n"
			s += "match.\n"
			s = fmt.Sprintf(s, "Add()", m.r, v.r)
			printErr(s)
		}
		if v.c != m.c {
			s := "\nIn %s, the number of the columns of the receiver is %d\n"
			s += "but the number of columns of the passed mat is %d. They must\n"
			s += "match.\n"
			s = fmt.Sprintf(s, "Add()", m.c, v.c)
			printErr(s)
		}
		vecf64.Add(m.vals, v.vals)
	default:
		s := "\nIn %s, the passed value must be a float64 or *Matf64.\n"
		s += "However, value of type  \"%v\" was received.\n"
		s = fmt.Sprintf(s, "Add()", reflect.TypeOf(v))
		printErr(s)
	}
	return m
}

/*
Sub carries the subtraction operation between each element of the receiver
and an object passed to it. Based on the type of the passed object, the results
of this method changes:

If the passed object is a float64, then it is subtracted from each element:

	m := matrix.Newf64(2, 3).SetAll(5.0)
	m.Sub(2.0)

This will result in all values of m being 3.0.
The passed Object can also be a Matf64, in which case each element of the passed
Matf64 is subtracted from the corresponding element of the receiver. Note
that the passed Matf64 must have the same shape as the receiver.

	m := matrix.Newf64(2, 3).SetAll(10.0)
	n := m.Copy()
	m.Sub(n)

This will result in each element of m being 0.0.
*/
func (m *Matf64) Sub(float64OrMatf64 interface{}) *Matf64 {
	switch v := float64OrMatf64.(type) {
	case float64:
		for i := range m.vals {
			m.vals[i] -= v
		}
	case *Matf64:
		if v.r != m.r {
			s := "\nIn %s, the number of the rows of the receiver is %d\n"
			s += "but the number of rows of the passed mat is %d. They must\n"
			s += "match.\n"
			s = fmt.Sprintf(s, "Sub()", m.r, v.r)
			printErr(s)
		}
		if v.c != m.c {
			s := "\nIn %s, the number of the columns of the receiver is %d\n"
			s += "but the number of columns of the passed mat is %d. They must\n"
			s += "match.\n"
			s = fmt.Sprintf(s, "Sub()", m.c, v.c)
			printErr(s)
		}
		vecf64.Sub(m.vals, v.vals)
	default:
		s := "\nIn %s, the passed value must be a float64 or *Matf64.\n"
		s += "However, value of type  \"%v\" was received.\n"
		s = fmt.Sprintf(s, "Sub()", reflect.TypeOf(v))
		printErr(s)
	}
	return m
}

/*
Div carries the division operation between each element of the receiver
and an object passed to it. Based on the type of the passed object, the results
of this method changes:

If the passed object is a float64, then each element of the receiver is devided
by it:

	m := Newf64(2, 3).SetAll(5.0)
	m.Div(2.0)

This will result in all values of m being 2.5. Note that the passed float64
cannot be 0.0.

The passed Object can also be a Matf64, in which case each element of the passed
Matf64 is subtracted from the corresponding element of the receiver. Note
that the passed Matf64 must have the same shape as the receiver, and it cannot
contains any elements which are 0.0.

	m := matrix.Newf64(2, 3).SetAll(10.0)
	n := m.Copy()
	m.Div(n)

This will result in each element of m being 1.0.
*/
func (m *Matf64) Div(float64OrMatf64 interface{}) *Matf64 {
	switch v := float64OrMatf64.(type) {
	case float64:
		for i := range m.vals {
			m.vals[i] /= v
		}
	case *Matf64:
		if v.r != m.r {
			s := "\nIn %s, the number of the rows of the receiver is %d\n"
			s += "but the number of rows of the passed mat is %d. They must\n"
			s += "match.\n"
			s = fmt.Sprintf(s, "Div()", m.r, v.r)
			printErr(s)
		}
		if v.c != m.c {
			s := "\nIn %s, the number of the columns of the receiver is %d\n"
			s += "but the number of columns of the passed mat is %d. They must\n"
			s += "match.\n"
			s = fmt.Sprintf(s, "Div()", m.c, v.c)
			printErr(s)
		}
		vecf64.Div(m.vals, v.vals)
	default:
		s := "\nIn %s, the passed value must be a float64 or *Matf64.\n"
		s += "However, value of type  \"%v\" was received.\n"
		s = fmt.Sprintf(s, "Div()", reflect.TypeOf(v))
		printErr(s)
	}
	return m
}

/*
Sum takes the sum of the elements of a Matf64. It can be called in one of two ways:

	m.Sum()

This will return the sum of all elements in m. This method can also be called by
passing 2 integers: 0 or 1 for row or column, and another int specifying the
row or column. For example:

	m.Sum(0, 2) // Returns the sum of the 3rd row
	m.Sum(1, 0) // Returns the sum of the first column.

Note that second passed integer cannot be less than 0, or greater that the
length of the matrix in that dimension.
*/
func (m *Matf64) Sum(args ...int) float64 {
	sum := 0.0
	switch len(args) {
	case 0:
		for i := range m.vals {
			sum += m.vals[i]
		}
	case 2:
		axis, slice := args[0], args[1]
		switch axis {
		case 0:
			if (slice >= m.r) || (slice < 0) {
				s := "\nIn %s the row %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Sum()", slice, m.r)
				printErr(s)
			}
			for i := 0; i < m.c; i++ {
				sum += m.vals[slice*m.c+i]
			}
		case 1:
			if (slice >= m.c) || (slice < 0) {
				s := "\nIn %s the column %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Sum()", slice, m.c)
				printErr(s)
			}
			for i := 0; i < m.r; i++ {
				sum += m.vals[i*m.c+slice]
			}
		default:
			s := "\nIn %s, the first argument must be 0 or 1, however %d "
			s += "was received.\n"
			s = fmt.Sprintf(s, "Sum()", axis)
			printErr(s)
		}
	default:
		s := "\nIn %s, 0 or 2 arguments expected, but %d was received.\n"
		s = fmt.Sprintf(s, "Sum()", len(args))
		printErr(s)
	}
	return sum
}

/*
Avg takes the average of the elements of a Matf64. It can be called in one of two ways:

	m.Avg()

This will return the average of all elements in m. This method can also be
called by passing 2 integers: 0 or 1 for row or column, and another int
specifying the row or column. For example:

	m.Avg(0, 2) // Returns the average of the 3rd row
	m.Avg(1, 0) // Returns the average of the first column.

Note that second passed integer cannot be less than 0, or greater that the
length of the matrix in that dimension.
*/
func (m *Matf64) Avg(args ...int) float64 {
	sum := 0.0
	switch len(args) {
	case 0:
		for i := range m.vals {
			sum += m.vals[i]
		}
		sum /= float64(len(m.vals))
	case 2:
		axis, slice := args[0], args[1]
		if axis == 0 {
			if (slice >= m.r) || (slice < 0) {
				s := "\nIn %s the row %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Avg()", slice, m.r)
				printErr(s)
			}
			for i := 0; i < m.c; i++ {
				sum += m.vals[slice*m.c+i]
			}
			sum /= float64(m.c)
		} else if axis == 1 {
			if (slice >= m.c) || (slice < 0) {
				s := "\nIn %s the column %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Avg()", slice, m.c)
				printErr(s)
			}
			for i := 0; i < m.r; i++ {
				sum += m.vals[i*m.c+slice]
			}
			sum /= float64(m.r)
		} else {
			s := "\nIn %s, the first argument must be 0 or 1, however %d "
			s += "was received.\n"
			s = fmt.Sprintf(s, "Avg()", axis)
			printErr(s)
		}
	default:
		s := "\nIn %s, 0 or 2 arguments expected, but %d was received.\n"
		s = fmt.Sprintf(s, "Avg()", len(args))
		printErr(s)
	}
	return sum
}

/*
Prd takes the product of the elements of a Matf64. It can be called in one of two
ways:

	m.Prd()

This will return the product of all elements in m. This method can also be
called by passing 2 integers: 0 or 1 for row or column, and another int
specifying the row or column. For example:

	m.Prd(0, 2) // Returns the product of the 3rd row
	m.Prd(1, 0) // Returns the product of the first column.

Note that second passed integer cannot be less than 0, or greater that the
length of the matrix in that dimension.
*/
func (m *Matf64) Prd(args ...int) float64 {
	prd := 1.0
	switch len(args) {
	case 0:
		for i := range m.vals {
			prd *= m.vals[i]
		}
	case 2:
		axis, slice := args[0], args[1]
		if axis == 0 {
			if (slice >= m.r) || (slice < 0) {
				s := "\nIn %s the row %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Prd()", slice, m.r)
				printErr(s)
			}
			for i := 0; i < m.c; i++ {
				prd *= m.vals[slice*m.c+i]
			}
		} else if axis == 1 {
			if (slice >= m.c) || (slice < 0) {
				s := "\nIn %s the column %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Prd()", slice, m.c)
				printErr(s)
			}
			for i := 0; i < m.r; i++ {
				prd *= m.vals[i*m.c+slice]
			}
		} else {
			s := "\nIn %s, the first argument must be 0 or 1, however %d "
			s += "was received.\n"
			s = fmt.Sprintf(s, "Prd()", axis)
			printErr(s)
		}
	default:
		s := "\nIn %s, 0 or 2 arguments expected, but %d was received.\n"
		s = fmt.Sprintf(s, "Prd()", len(args))
		printErr(s)
	}
	return prd
}

/*
Std takes the standard deviation of the elements of a Matf64. It can be called in
one of two ways:

	m.Std()

This will return the std. div. of all elements in m. This method can also be
called by passing 2 integers: 0 or 1 for row or column, and another int
specifying the row or column. For example:

	m.Std(0, 2) // Returns the standard deviation of the 3rd row
	m.Std(1, 0) // Returns the standard deviation of the first column.

Note that second passed integer cannot be less than 0, or greater that the
length of the matrix in that dimension.
*/
func (m *Matf64) Std(args ...int) float64 {
	std := 0.0
	switch len(args) {
	case 0:
		avg := m.Avg()
		sum := 0.0
		for i := range m.vals {
			sum += ((avg - m.vals[i]) * (avg - m.vals[i]))
		}
		std = math.Sqrt(sum / float64(len(m.vals)))
	case 2:
		axis, slice := args[0], args[1]
		if axis == 0 {
			if (slice >= m.r) || (slice < 0) {
				s := "\nIn %s the row %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Std()", slice, m.r)
				printErr(s)
			}
			avg := m.Avg(axis, slice)
			sum := 0.0
			for i := 0; i < m.c; i++ {
				sum += ((avg - m.vals[slice*m.c+i]) * (avg - m.vals[slice*m.c+i]))
			}
			std = math.Sqrt(sum / float64(len(m.vals)))
		} else if axis == 1 {
			if (slice >= m.c) || (slice < 0) {
				s := "\nIn %s the column %d is outside of bounds [0, %d)\n"
				s = fmt.Sprintf(s, "Std()", slice, m.c)
				printErr(s)
			}
			avg := m.Avg(axis, slice)
			sum := 0.0
			for i := 0; i < m.r; i++ {
				sum += ((avg - m.vals[i*m.c+slice]) * (avg - m.vals[i*m.c+slice]))
			}
			std = math.Sqrt(sum / float64(len(m.vals)))
		} else {
			s := "\nIn %s, the first argument must be 0 or 1, however %d "
			s += "was received.\n"
			s = fmt.Sprintf(s, "Std()", axis)
			printErr(s)
		}
	default:
		s := "\nIn %s, 0 or 2 arguments must be passed, but %d was received.\n"
		s = fmt.Sprintf(s, "Std()", len(args))
		printErr(s)
	}
	return std
}

/*
Dot is the matrix multiplication of two mat objects. Consider the following two
mats:

	m := matrix.Newf64(5, 6)
	n := matrix.Newf64(6, 10)

then

	o := m.Dot(n)

is a 5 by 10 mat whose element at row i and column j is given by:

	Sum(m.Row(i).Mul(n.col(j))
*/
func (m *Matf64) Dot(n *Matf64) *Matf64 {
	if m.c != n.r {
		s := "\nIn %s the number of columns of the first mat is %d\n"
		s += "which is not equal to the number of rows of the second mat,\n"
		s += "which is %d. They must be equal.\n"
		s = fmt.Sprintf(s, "Dot()", m.c, n.r)
		printErr(s)
	}
	o := Newf64(m.r, n.c)
	n.T()
	defer n.T()
	for i := 0; i < m.r; i++ {
		imc := i * m.c
		mrow := m.vals[imc : imc+m.c]
		for j := 0; j < n.r; j++ {
			jnc := j * n.c
			o.vals[i*n.r+j] = dotf64Helper(mrow, n.vals[jnc:jnc+n.c])
		}
	}
	return o
}

func dotf64Helper(a, b []float64) float64 {
	a = a[:len(a)]
	b = b[:len(a)]
	sum := 0.0
	for i, v := range a {
		sum += (v * b[i])
	}
	return sum
}

/*
String returns the string representation of a mat. This is done by putting
every row into a line, and separating the entries of that row by a space. note
that the last line does not contain a newline.
*/
func (m *Matf64) String() string {
	var str string
	str += "["
	for i := 0; i < m.r; i++ {
		for j := 0; j < m.c; j++ {
			if j == 0 {
				str += "["
			}
			str += strconv.FormatFloat(m.vals[i*m.c+j], 'f', 14, 64)
			if j+1 != m.c {
				str += ",\t"
			}
		}
		if i+1 <= m.r {
			str += "]\n "
		}
	}
	str = str[:len(str)-2] // take out the last newline.
	str += "]\n"
	return str
}

/*
AppendCol appends a column to the right side of a Matf64.
*/
func (m *Matf64) AppendCol(v []float64) *Matf64 {
	if m.r != len(v) {
		s := "\nIn %s the number of rows of the receiver is %d, while\n"
		s += "the number of rows of the vector is %d. They must be equal.\n"
		s = fmt.Sprintf(s, "AppendCol()", m.r, len(v))
		printErr(s)
	}
	// TODO: redo this by hand, instead of taking this shortcut... or check if
	// this is a huge bottleneck
	q := m.ToSlice2D()
	for i := range q {
		q[i] = append(q[i], v[i])
	}
	m.c++
	m.vals = append(m.vals, v...)
	for i := 0; i < m.r; i++ {
		for j := 0; j < m.c; j++ {
			m.vals[i*m.c+j] = q[i][j]
		}
	}
	return m
}

/*
AppendRow appends a row to the bottom of a Matf64.
*/
func (m *Matf64) AppendRow(v []float64) *Matf64 {
	if m.c != len(v) {
		s := "\nIn %s the number of cols of the receiver is %d, while\n"
		s += "the number of rows of the vector is %d. They must be equal.\n"
		s = fmt.Sprintf(s, "AppendRow()", m.c, len(v))
		printErr(s)
	}
	if cap(m.vals) < (len(m.vals) + len(v)) {
		newVals := make([]float64, len(m.vals)+len(v), len(m.vals)+len(v)*2)
		lastElem := len(m.vals)
		for i := range m.vals {
			newVals[i] = m.vals[i]
		}
		for i := range v {
			newVals[lastElem+i] = v[i]
		}
		m.vals = newVals
	} else {
		m.vals = append(m.vals, v...)
	}
	m.r++
	return m
}

/*
Concat merges a passed mat to the right side of the receiver. The passed mat
must therefore have the same number of rows as the receiver.
For example:

	m := matrix.Newf64(1, 2).SetAll(2.0) // [[2.0, 2.0]]
	n := matrix.Newf64(1, 3).SetAll(3.0) // [[3.0, 3.0, 3.0]]
	m.Concat(n)
	fmt.Println(m) // [[2.0, 2.0, 3.0, 3.0, 3.0]]

Note that in the current implementation this is a somewhat expensive function.
*/
func (m *Matf64) Concat(n *Matf64) *Matf64 {
	if m.r != n.r {
		s := "\nIn %s the number of rows of the receiver is %d, while\n"
		s += "the number of rows of the second Matf64 is %d. They must be equal.\n"
		s = fmt.Sprintf(s, "Concat()", m.r, n.r)
		printErr(s)
	}
	q := m.ToSlice2D()
	t := n.ToSlice1D()
	r := n.ToSlice2D()
	m.vals = append(m.vals, t...)
	for i := range q {
		q[i] = append(q[i], r[i]...)
	}
	m.c += n.c
	for i := 0; i < m.r; i++ {
		for j := 0; j < m.c; j++ {
			m.vals[i*m.c+j] = q[i][j]
		}
	}
	return m
}

/*
Append merges a passed mat to the botton of the receiver. The passed mat
must therefore have the same number of columns as the receiver.
For example:

	m := matrix.Newf64(1, 2).SetAll(2.0) // [[2.0, 2.0]]
	n := matrix.Newf64(2, 2).SetAll(3.0) // [[3.0, 3.0], [3.0, 3.0]]
	m.Append(n)
	fmt.Println(m) // [[2.0, 2.0], [3.0, 3.0], [3.0, 3.0]]

Note that in the current implementation this is a somewhat expensive function.
*/
func (m *Matf64) Append(n *Matf64) *Matf64 {
	if m.c != n.c {
		s := "\nIn %s the number of cols of the receiver is %d, while\n"
		s += "the number of cols of the passed Matf64 is %d. They must be equal.\n"
		s = fmt.Sprintf(s, "Append()", m.c, n.c)
		printErr(s)
	}
	m.vals = append(m.vals, n.vals...)
	return m
}
