[BugLab_Variable_Misuse]^RealMatrixImpl out = new RealMatrixImpl ( row, dimension ) ;^58^^^^^57^66^RealMatrixImpl out = new RealMatrixImpl ( dimension, dimension ) ;^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Variable_Misuse]^for  ( int dimension = 0; row < dimension; row++ )  {^60^^^^^57^66^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Variable_Misuse]^for  ( int row = 0; row < col; row++ )  {^60^^^^^57^66^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Argument_Swapping]^for  ( int d = 0; row < rowimension; row++ )  {^60^^^^^57^66^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Operator]^for  ( int row = 0; row == dimension; row++ )  {^60^^^^^57^66^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Operator]^for  ( int row = 0; row <= dimension; row++ )  {^60^^^^^57^66^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Literal]^for  ( int row = col; row < dimension; row++ )  {^60^^^^^57^66^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Literal]^for  ( int row = dimension; row < dimension; row++ )  {^60^^^^^57^66^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Variable_Misuse]^for  ( int row = 0; col < dimension; col++ )  {^61^^^^^57^66^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Variable_Misuse]^for  ( int col = 0; col < row; col++ )  {^61^^^^^57^66^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Argument_Swapping]^for  ( int col = 0; col < d; col++ )  {^61^^^^^57^66^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Operator]^for  ( int col = 0; col <= dimension; col++ )  {^61^^^^^57^66^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Literal]^for  ( int col = 1; col < dimension; col++ )  {^61^^^^^57^66^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Variable_Misuse]^d[row][col] = dimension == col ? 1d : 0d;^62^^^^^57^66^d[row][col] = row == col ? 1d : 0d;^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Variable_Misuse]^d[row][col] = row == row ? 1d : 0d;^62^^^^^57^66^d[row][col] = row == col ? 1d : 0d;^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Argument_Swapping]^d[row][col] = col == row ? 1d : 0d;^62^^^^^57^66^d[row][col] = row == col ? 1d : 0d;^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Operator]^d[row][col] = row > col ? 1d : 0d;^62^^^^^57^66^d[row][col] = row == col ? 1d : 0d;^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Variable_Misuse]^d[row][col] = row == dimension ? 1d : 0d;^62^^^^^57^66^d[row][col] = row == col ? 1d : 0d;^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Operator]^d[row][col] = row <= col ? 1d : 0d;^62^^^^^57^66^d[row][col] = row == col ? 1d : 0d;^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Argument_Swapping]^for  ( int dimension = 0; col < col; col++ )  {^61^^^^^57^66^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Argument_Swapping]^for  ( int d = 0; col < colimension; col++ )  {^61^^^^^57^66^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Operator]^d[row][col] = row != col ? 1d : 0d;^62^^^^^57^66^d[row][col] = row == col ? 1d : 0d;^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Wrong_Literal]^for  ( int col = dimension; col < dimension; col++ )  {^61^^^^^57^66^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createRealIdentityMatrix [RETURN_TYPE] RealMatrix   int dimension [VARIABLES] boolean  double[][]  d  RealMatrixImpl  out  int  col  dimension  row  
[BugLab_Argument_Swapping]^int nCols = rowData.length.length;^120^^^^^119^124^int nCols = rowData.length;^[CLASS] MatrixUtils  [METHOD] createRowRealMatrix [RETURN_TYPE] RealMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^int nCols = rowData;^120^^^^^119^124^int nCols = rowData.length;^[CLASS] MatrixUtils  [METHOD] createRowRealMatrix [RETURN_TYPE] RealMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Wrong_Literal]^double[][] data = new double[nCols][nCols];^121^^^^^119^124^double[][] data = new double[1][nCols];^[CLASS] MatrixUtils  [METHOD] createRowRealMatrix [RETURN_TYPE] RealMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^System.arraycopy ( nCols, 0, data[0], 0, rowData ) ;^122^^^^^119^124^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowRealMatrix [RETURN_TYPE] RealMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^System.arraycopy ( data, 0, rowData[0], 0, nCols ) ;^122^^^^^119^124^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowRealMatrix [RETURN_TYPE] RealMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^System.arraycopy ( rowData, 0, nCols[0], 0, data ) ;^122^^^^^119^124^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowRealMatrix [RETURN_TYPE] RealMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Wrong_Literal]^System.arraycopy ( rowData, nCols, data[nCols], nCols, nCols ) ;^122^^^^^119^124^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowRealMatrix [RETURN_TYPE] RealMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^int nCols = rowData.length.length;^136^^^^^135^140^int nCols = rowData.length;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^int nCols = rowData;^136^^^^^135^140^int nCols = rowData.length;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^System.arraycopy ( nCols, 0, data[0], 0, rowData ) ;^138^^^^^135^140^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^System.arraycopy ( rowData, 0, nCols[0], 0, data ) ;^138^^^^^135^140^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Wrong_Literal]^System.arraycopy ( rowData, nCols, data[nCols], nCols, nCols ) ;^138^^^^^135^140^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Wrong_Literal]^System.arraycopy ( rowData, -1, data[-1], -1, nCols ) ;^138^^^^^135^140^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   double[] rowData [VARIABLES] boolean  double[]  rowData  double[][]  data  int  nCols  
[BugLab_Argument_Swapping]^int nCols = rowData.length.length;^152^^^^^151^156^int nCols = rowData.length;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] rowData [VARIABLES] boolean  BigDecimal[]  rowData  int  nCols  BigDecimal[][]  data  
[BugLab_Argument_Swapping]^int nCols = rowData;^152^^^^^151^156^int nCols = rowData.length;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] rowData [VARIABLES] boolean  BigDecimal[]  rowData  int  nCols  BigDecimal[][]  data  
[BugLab_Argument_Swapping]^System.arraycopy ( nCols, 0, data[0], 0, rowData ) ;^154^^^^^151^156^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] rowData [VARIABLES] boolean  BigDecimal[]  rowData  int  nCols  BigDecimal[][]  data  
[BugLab_Argument_Swapping]^System.arraycopy ( data, 0, rowData[0], 0, nCols ) ;^154^^^^^151^156^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] rowData [VARIABLES] boolean  BigDecimal[]  rowData  int  nCols  BigDecimal[][]  data  
[BugLab_Wrong_Literal]^System.arraycopy ( rowData, nCols, data[nCols], nCols, nCols ) ;^154^^^^^151^156^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] rowData [VARIABLES] boolean  BigDecimal[]  rowData  int  nCols  BigDecimal[][]  data  
[BugLab_Wrong_Literal]^System.arraycopy ( rowData, -1, data[-1], -1, nCols ) ;^154^^^^^151^156^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] rowData [VARIABLES] boolean  BigDecimal[]  rowData  int  nCols  BigDecimal[][]  data  
[BugLab_Argument_Swapping]^int nCols = rowData.length.length;^168^^^^^167^172^int nCols = rowData.length;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   String[] rowData [VARIABLES] boolean  String[][]  data  String[]  rowData  int  nCols  
[BugLab_Argument_Swapping]^int nCols = rowData;^168^^^^^167^172^int nCols = rowData.length;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   String[] rowData [VARIABLES] boolean  String[][]  data  String[]  rowData  int  nCols  
[BugLab_Wrong_Literal]^String[][] data = new String[nCols][nCols];^169^^^^^167^172^String[][] data = new String[1][nCols];^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   String[] rowData [VARIABLES] boolean  String[][]  data  String[]  rowData  int  nCols  
[BugLab_Argument_Swapping]^System.arraycopy ( data, 0, rowData[0], 0, nCols ) ;^170^^^^^167^172^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   String[] rowData [VARIABLES] boolean  String[][]  data  String[]  rowData  int  nCols  
[BugLab_Argument_Swapping]^System.arraycopy ( nCols, 0, data[0], 0, rowData ) ;^170^^^^^167^172^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   String[] rowData [VARIABLES] boolean  String[][]  data  String[]  rowData  int  nCols  
[BugLab_Wrong_Literal]^System.arraycopy ( rowData, nCols, data[nCols], nCols, nCols ) ;^170^^^^^167^172^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   String[] rowData [VARIABLES] boolean  String[][]  data  String[]  rowData  int  nCols  
[BugLab_Wrong_Literal]^System.arraycopy ( rowData, 1, data[1], 1, nCols ) ;^170^^^^^167^172^System.arraycopy ( rowData, 0, data[0], 0, nCols ) ;^[CLASS] MatrixUtils  [METHOD] createRowBigMatrix [RETURN_TYPE] BigMatrix   String[] rowData [VARIABLES] boolean  String[][]  data  String[]  rowData  int  nCols  
[BugLab_Variable_Misuse]^int nRows = row;^184^^^^^183^190^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Argument_Swapping]^int nRows = columnData.length.length;^184^^^^^183^190^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Argument_Swapping]^int nRows = columnData;^184^^^^^183^190^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^double[][] data = new double[nRows][row];^185^^^^^183^190^double[][] data = new double[nRows][1];^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Argument_Swapping]^for  ( int nRows = 0; row < row; row++ )  {^186^^^^^183^190^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Operator]^for  ( int row = 0; row <= nRows; row++ )  {^186^^^^^183^190^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^for  ( int row = nRows; row < nRows; row++ )  {^186^^^^^183^190^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^for  ( int row = row; row < nRows; row++ )  {^186^^^^^183^190^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^data[row][-1] = columnData[row];^187^^^^^183^190^data[row][0] = columnData[row];^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^data[row][1] = columnData[row];^187^^^^^183^190^data[row][0] = columnData[row];^[CLASS] MatrixUtils  [METHOD] createColumnRealMatrix [RETURN_TYPE] RealMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Variable_Misuse]^int nRows = row;^202^^^^^201^208^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Argument_Swapping]^int nRows = columnData.length.length;^202^^^^^201^208^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Argument_Swapping]^int nRows = columnData;^202^^^^^201^208^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^double[][] data = new double[nRows][nRows];^203^^^^^201^208^double[][] data = new double[nRows][1];^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Variable_Misuse]^for  ( int nRows = 0; row < nRows; row++ )  {^204^^^^^201^208^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Argument_Swapping]^for  ( int nRows = 0; row < row; row++ )  {^204^^^^^201^208^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Operator]^for  ( int row = 0; row <= nRows; row++ )  {^204^^^^^201^208^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^for  ( int row = row; row < nRows; row++ )  {^204^^^^^201^208^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^for  ( int row = 1; row < nRows; row++ )  {^204^^^^^201^208^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Wrong_Literal]^data[row][nRows] = columnData[row];^205^^^^^201^208^data[row][0] = columnData[row];^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   double[] columnData [VARIABLES] boolean  double[]  columnData  double[][]  data  int  nRows  row  
[BugLab_Variable_Misuse]^int nRows = row;^220^^^^^219^226^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Argument_Swapping]^int nRows = columnData.length.length;^220^^^^^219^226^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Argument_Swapping]^int nRows = columnData;^220^^^^^219^226^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Wrong_Literal]^BigDecimal[][] data = new BigDecimal[nRows][nRows];^221^^^^^219^226^BigDecimal[][] data = new BigDecimal[nRows][1];^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Argument_Swapping]^for  ( int nRows = 0; row < row; row++ )  {^222^^^^^219^226^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Wrong_Operator]^for  ( int row = 0; row <= nRows; row++ )  {^222^^^^^219^226^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Wrong_Literal]^for  ( int row = nRows; row < nRows; row++ )  {^222^^^^^219^226^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Wrong_Literal]^data[row][nRows] = columnData[row];^223^^^^^219^226^data[row][0] = columnData[row];^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Wrong_Literal]^for  ( int row = -1; row < nRows; row++ )  {^222^^^^^219^226^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   BigDecimal[] columnData [VARIABLES] boolean  BigDecimal[]  columnData  int  nRows  row  BigDecimal[][]  data  
[BugLab_Variable_Misuse]^int nRows = row;^238^^^^^237^244^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   String[] columnData [VARIABLES] boolean  String[][]  data  String[]  columnData  int  nRows  row  
[BugLab_Argument_Swapping]^int nRows = columnData.length.length;^238^^^^^237^244^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   String[] columnData [VARIABLES] boolean  String[][]  data  String[]  columnData  int  nRows  row  
[BugLab_Argument_Swapping]^int nRows = columnData;^238^^^^^237^244^int nRows = columnData.length;^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   String[] columnData [VARIABLES] boolean  String[][]  data  String[]  columnData  int  nRows  row  
[BugLab_Variable_Misuse]^for  ( int nRows = 0; row < nRows; row++ )  {^240^^^^^237^244^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   String[] columnData [VARIABLES] boolean  String[][]  data  String[]  columnData  int  nRows  row  
[BugLab_Wrong_Operator]^for  ( int row = 0; row == nRows; row++ )  {^240^^^^^237^244^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   String[] columnData [VARIABLES] boolean  String[][]  data  String[]  columnData  int  nRows  row  
[BugLab_Wrong_Literal]^for  ( int row = row; row < nRows; row++ )  {^240^^^^^237^244^for  ( int row = 0; row < nRows; row++ )  {^[CLASS] MatrixUtils  [METHOD] createColumnBigMatrix [RETURN_TYPE] BigMatrix   String[] columnData [VARIABLES] boolean  String[][]  data  String[]  columnData  int  nRows  row  
[BugLab_Variable_Misuse]^BigMatrixImpl out = new BigMatrixImpl ( row, dimension ) ;^255^^^^^254^263^BigMatrixImpl out = new BigMatrixImpl ( dimension, dimension ) ;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Variable_Misuse]^for  ( int dimension = 0; row < dimension; row++ )  {^257^^^^^254^263^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Operator]^for  ( int row = 0; row <= dimension; row++ )  {^257^^^^^254^263^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Operator]^for  ( int row = 0; row == dimension; row++ )  {^257^^^^^254^263^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Literal]^for  ( int row = -1; row < dimension; row++ )  {^257^^^^^254^263^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Variable_Misuse]^for  ( int row = 0; col < dimension; col++ )  {^258^^^^^254^263^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Variable_Misuse]^for  ( int col = 0; col < row; col++ )  {^258^^^^^254^263^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Operator]^for  ( int col = 0; col <= dimension; col++ )  {^258^^^^^254^263^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Literal]^for  ( int col = col; col < dimension; col++ )  {^258^^^^^254^263^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Variable_Misuse]^d[row][col] = dimension == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Variable_Misuse]^d[row][col] = row == dimension ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Argument_Swapping]^d[row][col] = BigMatrixImpl.ZERO == col ? BigMatrixImpl.ONE : row;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Argument_Swapping]^d[row][col] = row == col ? BigMatrixImpl.ZERO : BigMatrixImpl.ONE;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Operator]^d[row][col] = row != col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Argument_Swapping]^d[row][col] = col == row ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Argument_Swapping]^d[row][col] = row == BigMatrixImpl.ONE ? col : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Literal]^for  ( int col = ; col < dimension; col++ )  {^258^^^^^254^263^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Argument_Swapping]^for  ( int dimension = 0; col < col; col++ )  {^258^^^^^254^263^for  ( int col = 0; col < dimension; col++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Variable_Misuse]^d[row][col] = row == row ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Operator]^d[row][col] = row >= col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Variable_Misuse]^d[row][col] = row == col ? this : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Argument_Swapping]^d[row][col] = row == BigMatrixImpl.ZERO ? BigMatrixImpl.ONE : col;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Argument_Swapping]^d[row][col] = BigMatrixImpl.ONE == col ? row : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Operator]^d[row][col] = row > col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^259^^^^^254^263^d[row][col] = row == col ? BigMatrixImpl.ONE : BigMatrixImpl.ZERO;^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  
[BugLab_Wrong_Literal]^for  ( int row = 1; row < dimension; row++ )  {^257^^^^^254^263^for  ( int row = 0; row < dimension; row++ )  {^[CLASS] MatrixUtils  [METHOD] createBigIdentityMatrix [RETURN_TYPE] BigMatrix   int dimension [VARIABLES] boolean  BigMatrixImpl  out  int  col  dimension  row  BigDecimal[][]  d  