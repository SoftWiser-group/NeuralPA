[BugLab_Variable_Misuse]^return seriesList.size (  ) ;^94^^^^^93^95^return this.seriesList.size (  ) ;^[CLASS] DefaultXYDataset  [METHOD] getSeriesCount [RETURN_TYPE] int   [VARIABLES] List  seriesKeys  seriesList  boolean  
[BugLab_Wrong_Operator]^if  (  ( series < 0 )  &&  ( series >= getSeriesCount (  )  )  )  {^109^^^^^108^113^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] DefaultXYDataset  [METHOD] getSeriesKey [RETURN_TYPE] Comparable   int series [VARIABLES] List  seriesKeys  seriesList  int  series  boolean  
[BugLab_Wrong_Operator]^if  (  ( series <= 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^109^^^^^108^113^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] DefaultXYDataset  [METHOD] getSeriesKey [RETURN_TYPE] Comparable   int series [VARIABLES] List  seriesKeys  seriesList  int  series  boolean  
[BugLab_Wrong_Operator]^if  (  ( series < 0 )  ||  ( series > getSeriesCount (  )  )  )  {^109^^^^^108^113^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] DefaultXYDataset  [METHOD] getSeriesKey [RETURN_TYPE] Comparable   int series [VARIABLES] List  seriesKeys  seriesList  int  series  boolean  
[BugLab_Wrong_Literal]^if  (  ( series < series )  ||  ( series >= getSeriesCount (  )  )  )  {^109^^^^^108^113^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] DefaultXYDataset  [METHOD] getSeriesKey [RETURN_TYPE] Comparable   int series [VARIABLES] List  seriesKeys  seriesList  int  series  boolean  
[BugLab_Variable_Misuse]^return  ( Comparable )  seriesList.get ( series ) ;^112^^^^^108^113^return  ( Comparable )  this.seriesKeys.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getSeriesKey [RETURN_TYPE] Comparable   int series [VARIABLES] List  seriesKeys  seriesList  int  series  boolean  
[BugLab_Argument_Swapping]^return  ( Comparable )  series.get ( this.seriesKeys ) ;^112^^^^^108^113^return  ( Comparable )  this.seriesKeys.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getSeriesKey [RETURN_TYPE] Comparable   int series [VARIABLES] List  seriesKeys  seriesList  int  series  boolean  
[BugLab_Variable_Misuse]^return seriesList.indexOf ( seriesKey ) ;^124^^^^^123^125^return this.seriesKeys.indexOf ( seriesKey ) ;^[CLASS] DefaultXYDataset  [METHOD] indexOf [RETURN_TYPE] int   Comparable seriesKey [VARIABLES] List  seriesKeys  seriesList  Comparable  seriesKey  boolean  
[BugLab_Argument_Swapping]^return seriesKey.indexOf ( this.seriesKeys ) ;^124^^^^^123^125^return this.seriesKeys.indexOf ( seriesKey ) ;^[CLASS] DefaultXYDataset  [METHOD] indexOf [RETURN_TYPE] int   Comparable seriesKey [VARIABLES] List  seriesKeys  seriesList  Comparable  seriesKey  boolean  
[BugLab_Wrong_Operator]^if  (  ( series < 0 )  &&  ( series >= getSeriesCount (  )  )  )  {^150^^^^^149^155^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] DefaultXYDataset  [METHOD] getItemCount [RETURN_TYPE] int   int series [VARIABLES] double[][]  seriesArray  List  seriesKeys  seriesList  boolean  int  series  
[BugLab_Wrong_Operator]^if  (  ( series <= 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^150^^^^^149^155^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] DefaultXYDataset  [METHOD] getItemCount [RETURN_TYPE] int   int series [VARIABLES] double[][]  seriesArray  List  seriesKeys  seriesList  boolean  int  series  
[BugLab_Wrong_Operator]^if  (  ( series < 0 )  ||  ( series > getSeriesCount (  )  )  )  {^150^^^^^149^155^if  (  ( series < 0 )  ||  ( series >= getSeriesCount (  )  )  )  {^[CLASS] DefaultXYDataset  [METHOD] getItemCount [RETURN_TYPE] int   int series [VARIABLES] double[][]  seriesArray  List  seriesKeys  seriesList  boolean  int  series  
[BugLab_Variable_Misuse]^double[][] seriesArray =  ( double[][] )  seriesList.get ( series ) ;^153^^^^^149^155^double[][] seriesArray =  ( double[][] )  this.seriesList.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getItemCount [RETURN_TYPE] int   int series [VARIABLES] double[][]  seriesArray  List  seriesKeys  seriesList  boolean  int  series  
[BugLab_Argument_Swapping]^double[][] this.seriesListArray =  ( double[][] )  series.get ( series ) ;^153^^^^^149^155^double[][] seriesArray =  ( double[][] )  this.seriesList.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getItemCount [RETURN_TYPE] int   int series [VARIABLES] double[][]  seriesArray  List  seriesKeys  seriesList  boolean  int  series  
[BugLab_Argument_Swapping]^return seriesArray[0].length[0].length;^154^^^^^149^155^return seriesArray[0].length;^[CLASS] DefaultXYDataset  [METHOD] getItemCount [RETURN_TYPE] int   int series [VARIABLES] double[][]  seriesArray  List  seriesKeys  seriesList  boolean  int  series  
[BugLab_Wrong_Literal]^return seriesArray[series].length;^154^^^^^149^155^return seriesArray[0].length;^[CLASS] DefaultXYDataset  [METHOD] getItemCount [RETURN_TYPE] int   int series [VARIABLES] double[][]  seriesArray  List  seriesKeys  seriesList  boolean  int  series  
[BugLab_Variable_Misuse]^double[][] itemData =  ( double[][] )  this.seriesList.get ( series ) ;^175^^^^^174^177^double[][] seriesData =  ( double[][] )  this.seriesList.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] double[][]  seriesData  List  seriesKeys  seriesList  boolean  int  item  series  
[BugLab_Variable_Misuse]^double[][] seriesData =  ( double[][] )  seriesList.get ( series ) ;^175^^^^^174^177^double[][] seriesData =  ( double[][] )  this.seriesList.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] double[][]  seriesData  List  seriesKeys  seriesList  boolean  int  item  series  
[BugLab_Argument_Swapping]^double[][] this.seriesListData =  ( double[][] )  series.get ( series ) ;^175^^^^^174^177^double[][] seriesData =  ( double[][] )  this.seriesList.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] double[][]  seriesData  List  seriesKeys  seriesList  boolean  int  item  series  
[BugLab_Wrong_Literal]^return seriesData[][item];^176^^^^^174^177^return seriesData[0][item];^[CLASS] DefaultXYDataset  [METHOD] getXValue [RETURN_TYPE] double   int series int item [VARIABLES] double[][]  seriesData  List  seriesKeys  seriesList  boolean  int  item  series  
[BugLab_Argument_Swapping]^return new Double ( getXValue ( item, series )  ) ;^197^^^^^196^198^return new Double ( getXValue ( series, item )  ) ;^[CLASS] DefaultXYDataset  [METHOD] getX [RETURN_TYPE] Number   int series int item [VARIABLES] List  seriesKeys  seriesList  int  item  series  boolean  
[BugLab_Variable_Misuse]^double[][] seriesData =  ( double[][] )  seriesList.get ( series ) ;^218^^^^^217^220^double[][] seriesData =  ( double[][] )  this.seriesList.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getYValue [RETURN_TYPE] double   int series int item [VARIABLES] double[][]  seriesData  List  seriesKeys  seriesList  boolean  int  item  series  
[BugLab_Argument_Swapping]^double[][] this.seriesListData =  ( double[][] )  series.get ( series ) ;^218^^^^^217^220^double[][] seriesData =  ( double[][] )  this.seriesList.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getYValue [RETURN_TYPE] double   int series int item [VARIABLES] double[][]  seriesData  List  seriesKeys  seriesList  boolean  int  item  series  
[BugLab_Variable_Misuse]^double[][] itemData =  ( double[][] )  this.seriesList.get ( series ) ;^218^^^^^217^220^double[][] seriesData =  ( double[][] )  this.seriesList.get ( series ) ;^[CLASS] DefaultXYDataset  [METHOD] getYValue [RETURN_TYPE] double   int series int item [VARIABLES] double[][]  seriesData  List  seriesKeys  seriesList  boolean  int  item  series  
[BugLab_Wrong_Literal]^return seriesData[series][item];^219^^^^^217^220^return seriesData[1][item];^[CLASS] DefaultXYDataset  [METHOD] getYValue [RETURN_TYPE] double   int series int item [VARIABLES] double[][]  seriesData  List  seriesKeys  seriesList  boolean  int  item  series  
[BugLab_Argument_Swapping]^return new Double ( getYValue ( item, series )  ) ;^240^^^^^239^241^return new Double ( getYValue ( series, item )  ) ;^[CLASS] DefaultXYDataset  [METHOD] getY [RETURN_TYPE] Number   int series int item [VARIABLES] List  seriesKeys  seriesList  int  item  series  boolean  
[BugLab_Wrong_Operator]^if  ( seriesKey != null )  {^254^^^^^253^279^if  ( seriesKey == null )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Operator]^if  ( data != null )  {^258^^^^^253^279^if  ( data == null )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Variable_Misuse]^if  ( seriesIndex != 2 )  {^261^^^^^253^279^if  ( data.length != 2 )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Argument_Swapping]^if  ( data.length.length != 2 )  {^261^^^^^253^279^if  ( data.length != 2 )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Argument_Swapping]^if  ( data != 2 )  {^261^^^^^253^279^if  ( data.length != 2 )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Operator]^if  ( data.length == 2 )  {^261^^^^^253^279^if  ( data.length != 2 )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Literal]^if  ( data.length != 3 )  {^261^^^^^253^279^if  ( data.length != 2 )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Argument_Swapping]^if  ( data[0].length[0].length != data[1].length )  {^265^^^^^253^279^if  ( data[0].length != data[1].length )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Operator]^if  ( data[0].length == data[1].length )  {^265^^^^^253^279^if  ( data[0].length != data[1].length )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Literal]^if  ( data[seriesIndex].length != data[1].length )  {^265^^^^^253^279^if  ( data[0].length != data[1].length )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Operator]^if  ( seriesIndex <= -1 )  {^270^^^^^253^279^if  ( seriesIndex == -1 )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Literal]^if  ( seriesIndex == -0 )  {^270^^^^^253^279^if  ( seriesIndex == -1 )  {^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Argument_Swapping]^this.seriesList.add ( data, seriesIndex ) ;^276^^^^^253^279^this.seriesList.add ( seriesIndex, data ) ;^[CLASS] DefaultXYDataset  [METHOD] addSeries [RETURN_TYPE] void   Comparable seriesKey double[][] data [VARIABLES] double[][]  data  List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Operator]^if  ( seriesIndex < 0 )  {^290^^^^^288^295^if  ( seriesIndex >= 0 )  {^[CLASS] DefaultXYDataset  [METHOD] removeSeries [RETURN_TYPE] void   Comparable seriesKey [VARIABLES] List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Literal]^if  ( seriesIndex >= seriesIndex )  {^290^^^^^288^295^if  ( seriesIndex >= 0 )  {^[CLASS] DefaultXYDataset  [METHOD] removeSeries [RETURN_TYPE] void   Comparable seriesKey [VARIABLES] List  seriesKeys  seriesList  Comparable  seriesKey  boolean  int  seriesIndex  
[BugLab_Wrong_Operator]^if  ( obj != this )  {^313^^^^^312^338^if  ( obj == this )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^return false;^314^^^^^312^338^return true;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Operator]^if  ( ! ( obj  ^  DefaultXYDataset )  )  {^316^^^^^312^338^if  ( ! ( obj instanceof DefaultXYDataset )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^return true;^317^^^^^312^338^return false;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^if  ( !this.seriesKeys.equals ( seriesList )  )  {^320^^^^^312^338^if  ( !this.seriesKeys.equals ( that.seriesKeys )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Argument_Swapping]^if  ( !this.seriesKeys.equals ( that )  )  {^320^^^^^312^338^if  ( !this.seriesKeys.equals ( that.seriesKeys )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^return true;^321^^^^^312^338^return false;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^if  ( !Arrays.equals ( d2y, d2x )  )  {^328^^^^^312^338^if  ( !Arrays.equals ( d1x, d2x )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^if  ( !Arrays.equals ( d1x, d2y )  )  {^328^^^^^312^338^if  ( !Arrays.equals ( d1x, d2x )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Argument_Swapping]^if  ( !Arrays.equals ( d2x, d1x )  )  {^328^^^^^312^338^if  ( !Arrays.equals ( d1x, d2x )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^return true;^329^^^^^312^338^return false;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^if  ( !Arrays.equals ( d2x, d2y )  )  {^333^^^^^312^338^if  ( !Arrays.equals ( d1y, d2y )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^if  ( !Arrays.equals ( d1y, d2x )  )  {^333^^^^^312^338^if  ( !Arrays.equals ( d1y, d2y )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Argument_Swapping]^if  ( !Arrays.equals ( d2y, d1y )  )  {^333^^^^^312^338^if  ( !Arrays.equals ( d1y, d2y )  )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^return true;^334^^^^^312^338^return false;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i > this.seriesList.size (  ) ; i++ )  {^323^^^^^312^338^for  ( int i = 0; i < this.seriesList.size (  ) ; i++ )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < this.seriesList.size (  ) ; i++ )  {^323^^^^^312^338^for  ( int i = 0; i < this.seriesList.size (  ) ; i++ )  {^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^double[][] d1 =  ( double[][] )  seriesList.get ( i ) ;^324^^^^^312^338^double[][] d1 =  ( double[][] )  this.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Argument_Swapping]^double[][] d1 =  ( double[][] )  i.get ( this.seriesList ) ;^324^^^^^312^338^double[][] d1 =  ( double[][] )  this.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^double[][] d2 =  ( double[][] )  seriesList.get ( i ) ;^325^^^^^312^338^double[][] d2 =  ( double[][] )  that.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Argument_Swapping]^double[][] d2 =  ( double[][] )  that.seriesList.seriesList.get ( i ) ;^325^^^^^312^338^double[][] d2 =  ( double[][] )  that.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Argument_Swapping]^double[][] d2 =  ( double[][] )  i.seriesList.get ( that ) ;^325^^^^^312^338^double[][] d2 =  ( double[][] )  that.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Argument_Swapping]^double[][] d2 =  ( double[][] )  i.get ( that.seriesList ) ;^325^^^^^312^338^double[][] d2 =  ( double[][] )  that.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^double[] d2x = d1[0];^326^^^^^312^338^double[] d1x = d1[0];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^double[] d1x = d1[i];^326^^^^^312^338^double[] d1x = d1[0];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^double[] d1x = d2[0];^327^^^^^312^338^double[] d2x = d2[0];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^double[] d2x = d2[-1];^327^^^^^312^338^double[] d2x = d2[0];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^double[] d2y = d1[1];^331^^^^^312^338^double[] d1y = d1[1];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^double[] d1y = d2[1];^332^^^^^312^338^double[] d2y = d2[1];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^double[] d2y = d2[0];^332^^^^^312^338^double[] d2y = d2[1];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Argument_Swapping]^double[][] d2 =  ( double[][] )  that.get ( i ) ;^325^^^^^312^338^double[][] d2 =  ( double[][] )  that.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^double[] d2x = d2[i];^327^^^^^312^338^double[] d2x = d2[0];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^double[] d2y = d2[i];^332^^^^^312^338^double[] d2y = d2[1];^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Wrong_Literal]^return false;^337^^^^^312^338^return true;^[CLASS] DefaultXYDataset  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] boolean  double[]  d1x  d1y  d2x  d2y  DefaultXYDataset  that  double[][]  d1  d2  List  seriesKeys  seriesList  Object  obj  int  i  
[BugLab_Variable_Misuse]^result = seriesList.hashCode (  ) ;^347^^^^^345^350^result = this.seriesKeys.hashCode (  ) ;^[CLASS] DefaultXYDataset  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] List  seriesKeys  seriesList  int  result  boolean  
[BugLab_Variable_Misuse]^result = 29 * result + seriesList.hashCode (  ) ;^348^^^^^345^350^result = 29 * result + this.seriesList.hashCode (  ) ;^[CLASS] DefaultXYDataset  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] List  seriesKeys  seriesList  int  result  boolean  
[BugLab_Argument_Swapping]^result = 29 * this.seriesList + result.hashCode (  ) ;^348^^^^^345^350^result = 29 * result + this.seriesList.hashCode (  ) ;^[CLASS] DefaultXYDataset  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] List  seriesKeys  seriesList  int  result  boolean  
[BugLab_Wrong_Operator]^result = 29 * result + this.seriesList.hashCode (  !=  ) ;^348^^^^^345^350^result = 29 * result + this.seriesList.hashCode (  ) ;^[CLASS] DefaultXYDataset  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] List  seriesKeys  seriesList  int  result  boolean  
[BugLab_Wrong_Operator]^result = 29 - result + this.seriesList.hashCode (  ) ;^348^^^^^345^350^result = 29 * result + this.seriesList.hashCode (  ) ;^[CLASS] DefaultXYDataset  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] List  seriesKeys  seriesList  int  result  boolean  
[BugLab_Wrong_Literal]^result =  * result + this.seriesList.hashCode (  ) ;^348^^^^^345^350^result = 29 * result + this.seriesList.hashCode (  ) ;^[CLASS] DefaultXYDataset  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] List  seriesKeys  seriesList  int  result  boolean  
[BugLab_Variable_Misuse]^clone.seriesKeys = new java.util.ArrayList ( seriesList ) ;^363^^^^^361^376^clone.seriesKeys = new java.util.ArrayList ( this.seriesKeys ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^clone.seriesList = new ArrayList ( seriesList.size (  )  ) ;^364^^^^^361^376^clone.seriesList = new ArrayList ( this.seriesList.size (  )  ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < seriesList.size (  ) ; i++ )  {^365^^^^^361^376^for  ( int i = 0; i < this.seriesList.size (  ) ; i++ )  {^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= this.seriesList.size (  ) ; i++ )  {^365^^^^^361^376^for  ( int i = 0; i < this.seriesList.size (  ) ; i++ )  {^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < this.seriesList.size (  ) ; i++ )  {^365^^^^^361^376^for  ( int i = 0; i < this.seriesList.size (  ) ; i++ )  {^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < this.seriesList.size (  ) ; i++ )  {^365^^^^^361^376^for  ( int i = 0; i < this.seriesList.size (  ) ; i++ )  {^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < this.seriesList.size (  ) ; i++ )  {^365^^^^^361^376^for  ( int i = 0; i < this.seriesList.size (  ) ; i++ )  {^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^double[][] data =  ( double[][] )  seriesList.get ( i ) ;^366^^^^^361^376^double[][] data =  ( double[][] )  this.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Argument_Swapping]^double[][] data =  ( double[][] )  i.get ( this.seriesList ) ;^366^^^^^361^376^double[][] data =  ( double[][] )  this.seriesList.get ( i ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^double[] x = data[i];^367^^^^^361^376^double[] x = data[0];^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^double[] y = data[i];^368^^^^^361^376^double[] y = data[1];^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^double[] yyx = new double[x.length];^369^^^^^361^376^double[] xx = new double[x.length];^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^double[] xxy = new double[y.length];^370^^^^^361^376^double[] yy = new double[y.length];^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^System.arraycopy ( yy, 0, xx, 0, x.length ) ;^371^^^^^361^376^System.arraycopy ( x, 0, xx, 0, x.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^System.arraycopy ( x, 0, yy, 0, x.length ) ;^371^^^^^361^376^System.arraycopy ( x, 0, xx, 0, x.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^System.arraycopy ( x, 0, xx, 0, i ) ;^371^^^^^361^376^System.arraycopy ( x, 0, xx, 0, x.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Argument_Swapping]^System.arraycopy ( x.length, 0, xx, 0, x ) ;^371^^^^^361^376^System.arraycopy ( x, 0, xx, 0, x.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Argument_Swapping]^System.arraycopy ( xx, 0, x, 0, x.length ) ;^371^^^^^361^376^System.arraycopy ( x, 0, xx, 0, x.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^System.arraycopy ( x, i, xx, i, x.length ) ;^371^^^^^361^376^System.arraycopy ( x, 0, xx, 0, x.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^System.arraycopy ( y, 0, xx, 0, y.length ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^System.arraycopy ( y, 0, yy, 0, i ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Argument_Swapping]^System.arraycopy ( yy, 0, y, 0, y.length ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Argument_Swapping]^System.arraycopy ( y.length, 0, yy, 0, y ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^System.arraycopy ( y, -1, yy, -1, y.length ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^System.arraycopy ( y, i, yy, i, y.length ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^clone.seriesList.add ( i, new double[][] {xx, y} ) ;^373^^^^^361^376^clone.seriesList.add ( i, new double[][] {xx, yy} ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^System.arraycopy ( x, -1, xx, -1, x.length ) ;^371^^^^^361^376^System.arraycopy ( x, 0, xx, 0, x.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^System.arraycopy ( yy, 0, yy, 0, y.length ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Variable_Misuse]^System.arraycopy ( y, 0, y, 0, y.length ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Argument_Swapping]^System.arraycopy ( y, 0, y.length, 0, yy ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  
[BugLab_Wrong_Literal]^System.arraycopy ( y, , yy, , y.length ) ;^372^^^^^361^376^System.arraycopy ( y, 0, yy, 0, y.length ) ;^[CLASS] DefaultXYDataset  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] double[][]  data  List  seriesKeys  seriesList  boolean  double[]  x  xx  y  yy  DefaultXYDataset  clone  int  i  