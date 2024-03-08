[P1_Replace_Type]^private static final  short  serialVersionUID = 4630760091523940820L;^62^^^^^57^67^private static final long serialVersionUID = 4630760091523940820L;^[CLASS] StandardCategorySeriesLabelGenerator   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID = 4630760091523940820;^62^^^^^57^67^private static final long serialVersionUID = 4630760091523940820L;^[CLASS] StandardCategorySeriesLabelGenerator   [VARIABLES] 
[P1_Replace_Type]^public static final char DEFAULT_LABEL_FORMAT = "{0}";^65^^^^^60^70^public static final String DEFAULT_LABEL_FORMAT = "{0}";^[CLASS] StandardCategorySeriesLabelGenerator   [VARIABLES] 
[P3_Replace_Literal]^public static final String DEFAULT_LABEL_FORMAT = "{0}{";^65^^^^^60^70^public static final String DEFAULT_LABEL_FORMAT = "{0}";^[CLASS] StandardCategorySeriesLabelGenerator   [VARIABLES] 
[P1_Replace_Type]^private char formatPattern;^68^^^^^63^73^private String formatPattern;^[CLASS] StandardCategorySeriesLabelGenerator   [VARIABLES] 
[P5_Replace_Variable]^this ( formatPattern ) ;^75^^^^^74^76^this ( DEFAULT_LABEL_FORMAT ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] StandardCategorySeriesLabelGenerator()   [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  formatPattern  boolean  
[P14_Delete_Statement]^^75^^^^^74^76^this ( DEFAULT_LABEL_FORMAT ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] StandardCategorySeriesLabelGenerator()   [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  formatPattern  boolean  
[P2_Replace_Operator]^if  ( format != null )  {^84^^^^^83^88^if  ( format == null )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P5_Replace_Variable]^if  ( formatPattern == null )  {^84^^^^^83^88^if  ( format == null )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P8_Replace_Mix]^if  ( format == true )  {^84^^^^^83^88^if  ( format == null )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P9_Replace_Statement]^if  ( dataset == null )  {^84^^^^^83^88^if  ( format == null )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Null 'format' argument.");^84^85^86^^^83^88^if  ( format == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P16_Remove_Block]^^84^85^86^^^83^88^if  ( format == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P13_Insert_Block]^if  ( dataset == null )  {     throw new IllegalArgumentException ( "Null 'dataset' argument." ) ; }^84^^^^^83^88^[Delete]^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P4_Replace_Constructor]^throw throw  new IllegalArgumentException ( "Null 'dataset' argument." )   ;^85^^^^^83^88^throw new IllegalArgumentException  (" ")  ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P13_Insert_Block]^if  ( dataset == null )  {     throw new IllegalArgumentException ( "Null 'dataset' argument." ) ; }^85^^^^^83^88^[Delete]^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P13_Insert_Block]^if  ( format == null )  {     throw new IllegalArgumentException ( "Null 'format' argument." ) ; }^85^^^^^83^88^[Delete]^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P5_Replace_Variable]^this.formatPattern = formatPattern;^87^^^^^83^88^this.formatPattern = format;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P2_Replace_Operator]^if  ( dataset != null )  {^99^^^^^98^105^if  ( dataset == null )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P8_Replace_Mix]^if  ( dataset == true )  {^99^^^^^98^105^if  ( dataset == null )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P9_Replace_Statement]^if  ( format == null )  {^99^^^^^98^105^if  ( dataset == null )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Null 'dataset' argument.");^99^100^101^^^98^105^if  ( dataset == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P16_Remove_Block]^^99^100^101^^^98^105^if  ( dataset == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P13_Insert_Block]^if  ( format == null )  {     throw new IllegalArgumentException ( "Null 'format' argument." ) ; }^99^^^^^98^105^[Delete]^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P4_Replace_Constructor]^throw throw  new IllegalArgumentException ( "Null 'format' argument." )   ;^100^^^^^98^105^throw new IllegalArgumentException  (" ")  ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P13_Insert_Block]^if  ( dataset == null )  {     throw new IllegalArgumentException ( "Null 'dataset' argument." ) ; }^100^^^^^98^105^[Delete]^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P13_Insert_Block]^if  ( format == null )  {     throw new IllegalArgumentException ( "Null 'format' argument." ) ; }^100^^^^^98^105^[Delete]^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P1_Replace_Type]^char label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^String label = MessageFormat.format ( formatPattern, createItemArray ( dataset, series )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^String label = MessageFormat.format ( this.formatPattern, createItemArray (  series )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^String label = MessageFormat.format (  createItemArray ( dataset, series )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^String label = MessageFormat.format ( this.formatPattern, createItemArray ( series, dataset )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^String label = MessageFormat.format ( series, createItemArray ( dataset, this.formatPattern )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P7_Replace_Invocation]^String label = MessageFormat.format ( this.formatPattern, generateLabel ( dataset, series )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^String label = MessageFormat.format ( dataset, createItemArray ( this.formatPattern, series )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P8_Replace_Mix]^String label = MessageFormat.format ( label, createItemArray ( dataset, series )  ) ;^102^103^^^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P14_Delete_Statement]^^102^103^104^105^^98^105^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series )  ) ; return label; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^createItemArray (  series )  ) ;^103^^^^^98^105^createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^createItemArray ( dataset )  ) ;^103^^^^^98^105^createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^createItemArray ( series, dataset )  ) ;^103^^^^^98^105^createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P7_Replace_Invocation]^generateLabel ( dataset, series )  ) ;^103^^^^^98^105^createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P14_Delete_Statement]^^103^^^^^98^105^createItemArray ( dataset, series )  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P5_Replace_Variable]^return formatPattern;^104^^^^^98^105^return label;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  CategoryDataset  dataset  
[P3_Replace_Literal]^Object[] result = new Object[series];^117^^^^^116^120^Object[] result = new Object[1];^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  CategoryDataset  dataset  
[P3_Replace_Literal]^result[series] = dataset.getRowKey ( series ) .toString (  ) ;^118^^^^^116^120^result[0] = dataset.getRowKey ( series ) .toString (  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  CategoryDataset  dataset  
[P5_Replace_Variable]^result[0] = series.getRowKey ( dataset ) .toString (  ) ;^118^^^^^116^120^result[0] = dataset.getRowKey ( series ) .toString (  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  CategoryDataset  dataset  
[P8_Replace_Mix]^result[1] = dataset.getRowKey ( series ) .toString (  ) ;^118^^^^^116^120^result[0] = dataset.getRowKey ( series ) .toString (  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  CategoryDataset  dataset  
[P14_Delete_Statement]^^118^119^^^^116^120^result[0] = dataset.getRowKey ( series ) .toString (  ) ; return result;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   CategoryDataset dataset int series [VARIABLES] String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  CategoryDataset  dataset  
[P8_Replace_Mix]^return super .clone (  )  ;^130^^^^^129^131^return super.clone (  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  
[P14_Delete_Statement]^^130^^^^^129^131^return super.clone (  ) ;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  
[P2_Replace_Operator]^if  ( obj != this )  {^141^^^^^140^153^if  ( obj == this )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return true;^141^142^143^^^140^153^if  ( obj == this )  { return true; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P16_Remove_Block]^^141^142^143^^^140^153^if  ( obj == this )  { return true; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return false;^142^^^^^140^153^return true;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P2_Replace_Operator]^if  ( ! ( obj  ||  StandardCategorySeriesLabelGenerator )  )  {^144^^^^^140^153^if  ( ! ( obj instanceof StandardCategorySeriesLabelGenerator )  )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return false;^144^145^146^^^140^153^if  ( ! ( obj instanceof StandardCategorySeriesLabelGenerator )  )  { return false; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P16_Remove_Block]^^144^145^146^^^140^153^if  ( ! ( obj instanceof StandardCategorySeriesLabelGenerator )  )  { return false; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return true;^145^^^^^140^153^return false;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( !this.formatPattern.equals ( label )  )  {^149^^^^^140^153^if  ( !this.formatPattern.equals ( that.formatPattern )  )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( !this.formatPattern.equals ( that.formatPattern.formatPattern )  )  {^149^^^^^140^153^if  ( !this.formatPattern.equals ( that.formatPattern )  )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( !this.formatPattern.equals ( that )  )  {^149^^^^^140^153^if  ( !this.formatPattern.equals ( that.formatPattern )  )  {^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return false;^149^150^151^^^140^153^if  ( !this.formatPattern.equals ( that.formatPattern )  )  { return false; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P16_Remove_Block]^^149^150^151^^^140^153^if  ( !this.formatPattern.equals ( that.formatPattern )  )  { return false; }^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return true;^150^^^^^140^153^return false;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( equals ( that.formatPattern )  )  )  {     return false; }^149^^^^^140^153^[Delete]^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return false;^152^^^^^140^153^return true;^[CLASS] StandardCategorySeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  StandardCategorySeriesLabelGenerator  that  boolean  long  serialVersionUID  