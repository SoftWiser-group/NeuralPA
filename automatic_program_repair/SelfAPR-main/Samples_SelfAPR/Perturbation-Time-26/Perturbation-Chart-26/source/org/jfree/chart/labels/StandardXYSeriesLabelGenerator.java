[P1_Replace_Type]^private static final  int  serialVersionUID = 1916017081848400024L;^67^^^^^62^72^private static final long serialVersionUID = 1916017081848400024L;^[CLASS] StandardXYSeriesLabelGenerator   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID  = null ;^67^^^^^62^72^private static final long serialVersionUID = 1916017081848400024L;^[CLASS] StandardXYSeriesLabelGenerator   [VARIABLES] 
[P1_Replace_Type]^public static final char DEFAULT_LABEL_FORMAT = "{0}";^70^^^^^65^75^public static final String DEFAULT_LABEL_FORMAT = "{0}";^[CLASS] StandardXYSeriesLabelGenerator   [VARIABLES] 
[P3_Replace_Literal]^public static final String DEFAULT_LABEL_FORMAT = "0";^70^^^^^65^75^public static final String DEFAULT_LABEL_FORMAT = "{0}";^[CLASS] StandardXYSeriesLabelGenerator   [VARIABLES] 
[P8_Replace_Mix]^public static final String DEFAULT_LABEL_FORMAT ;^70^^^^^65^75^public static final String DEFAULT_LABEL_FORMAT = "{0}";^[CLASS] StandardXYSeriesLabelGenerator   [VARIABLES] 
[P1_Replace_Type]^private char formatPattern;^73^^^^^68^78^private String formatPattern;^[CLASS] StandardXYSeriesLabelGenerator   [VARIABLES] 
[P8_Replace_Mix]^this ( formatPattern ) ;^80^^^^^79^81^this ( DEFAULT_LABEL_FORMAT ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] StandardXYSeriesLabelGenerator()   [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  formatPattern  boolean  
[P14_Delete_Statement]^^80^^^^^79^81^this ( DEFAULT_LABEL_FORMAT ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] StandardXYSeriesLabelGenerator()   [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  formatPattern  boolean  
[P2_Replace_Operator]^if  ( format != null )  {^89^^^^^88^93^if  ( format == null )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P5_Replace_Variable]^if  ( formatPattern == null )  {^89^^^^^88^93^if  ( format == null )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P8_Replace_Mix]^if  ( format == true )  {^89^^^^^88^93^if  ( format == null )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P9_Replace_Statement]^if  ( dataset == null )  {^89^^^^^88^93^if  ( format == null )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Null 'format' argument.");^89^90^91^^^88^93^if  ( format == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P16_Remove_Block]^^89^90^91^^^88^93^if  ( format == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P13_Insert_Block]^if  ( dataset == null )  {     throw new IllegalArgumentException ( "Null 'dataset' argument." ) ; }^89^^^^^88^93^[Delete]^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P4_Replace_Constructor]^throw throw  new IllegalArgumentException ( "Null 'dataset' argument." )   ;^90^^^^^88^93^throw new IllegalArgumentException  (" ")  ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P13_Insert_Block]^if  ( dataset == null )  {     throw new IllegalArgumentException ( "Null 'dataset' argument." ) ; }^90^^^^^88^93^[Delete]^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P13_Insert_Block]^if  ( format == null )  {     throw new IllegalArgumentException ( "Null 'format' argument." ) ; }^90^^^^^88^93^[Delete]^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P8_Replace_Mix]^return ;^90^^^^^88^93^throw new IllegalArgumentException  (" ")  ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P5_Replace_Variable]^this.formatPattern = formatPattern;^92^^^^^88^93^this.formatPattern = format;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] <init> [RETURN_TYPE] String)   String format [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  boolean  
[P2_Replace_Operator]^if  ( dataset != null )  {^105^^^^^104^112^if  ( dataset == null )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P8_Replace_Mix]^if  ( dataset == false )  {^105^^^^^104^112^if  ( dataset == null )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P9_Replace_Statement]^if  ( format == null )  {^105^^^^^104^112^if  ( dataset == null )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Null 'dataset' argument.");^105^106^107^^^104^112^if  ( dataset == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P16_Remove_Block]^^105^106^107^^^104^112^if  ( dataset == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P13_Insert_Block]^if  ( format == null )  {     throw new IllegalArgumentException ( "Null 'format' argument." ) ; }^105^^^^^104^112^[Delete]^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P4_Replace_Constructor]^throw throw  new IllegalArgumentException ( "Null 'format' argument." )   ;^106^^^^^104^112^throw new IllegalArgumentException  (" ")  ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P13_Insert_Block]^if  ( dataset == null )  {     throw new IllegalArgumentException ( "Null 'dataset' argument." ) ; }^106^^^^^104^112^[Delete]^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P13_Insert_Block]^if  ( format == null )  {     throw new IllegalArgumentException ( "Null 'format' argument." ) ; }^106^^^^^104^112^[Delete]^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P8_Replace_Mix]^return ;^106^^^^^104^112^throw new IllegalArgumentException  (" ")  ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P1_Replace_Type]^char label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^String label = MessageFormat.format ( formatPattern, createItemArray ( dataset, series ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^String label = MessageFormat.format ( this.formatPattern, createItemArray (  series ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^String label = MessageFormat.format (  createItemArray ( dataset, series ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^String label = MessageFormat.format ( this.formatPattern, createItemArray ( series, dataset ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^String label = MessageFormat.format ( series, createItemArray ( dataset, this.formatPattern ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P7_Replace_Invocation]^String label = MessageFormat.format ( this.formatPattern, generateLabel ( dataset, series ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^String label = MessageFormat.format ( label, createItemArray ( dataset, series ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^String label = MessageFormat.format ( dataset, createItemArray ( this.formatPattern, series ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P8_Replace_Mix]^String label = MessageFormat.format ( label, generateLabel ( dataset, series ) ) ;^108^109^110^^^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P14_Delete_Statement]^^108^109^110^111^112^104^112^String label = MessageFormat.format ( this.formatPattern, createItemArray ( dataset, series ) ) ; return label; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^this.formatPattern, createItemArray (  series ) ) ;^109^110^^^^104^112^this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^this.formatPattern, createItemArray ( dataset ) ) ;^109^110^^^^104^112^this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^this.formatPattern, createItemArray ( series, dataset ) ) ;^109^110^^^^104^112^this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P7_Replace_Invocation]^this.formatPattern, generateLabel ( dataset, series ) ) ;^109^110^^^^104^112^this.formatPattern, createItemArray ( dataset, series ) ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P14_Delete_Statement]^^109^110^111^112^^104^112^this.formatPattern, createItemArray ( dataset, series ) ) ; return label; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P5_Replace_Variable]^return formatPattern;^111^^^^^104^112^return label;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] generateLabel [RETURN_TYPE] String   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  
[P3_Replace_Literal]^Object[] result = new Object[4];^124^^^^^123^127^Object[] result = new Object[1];^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  
[P3_Replace_Literal]^result[series] = dataset.getSeriesKey ( series ) .toString (  ) ;^125^^^^^123^127^result[0] = dataset.getSeriesKey ( series ) .toString (  ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  
[P8_Replace_Mix]^result[3] = dataset.getSeriesKey ( series ) .toString (  ) ;^125^^^^^123^127^result[0] = dataset.getSeriesKey ( series ) .toString (  ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  
[P5_Replace_Variable]^result[0] = series.getSeriesKey ( dataset ) .toString (  ) ;^125^^^^^123^127^result[0] = dataset.getSeriesKey ( series ) .toString (  ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  
[P14_Delete_Statement]^^125^^^^^123^127^result[0] = dataset.getSeriesKey ( series ) .toString (  ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] createItemArray [RETURN_TYPE] Object[]   XYDataset dataset int series [VARIABLES] XYDataset  dataset  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  int  series  Object[]  result  
[P7_Replace_Invocation]^return super .clone (  )  ;^139^^^^^138^140^return super.clone (  ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  
[P14_Delete_Statement]^^139^^^^^138^140^return super.clone (  ) ;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] long  serialVersionUID  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  
[P2_Replace_Operator]^if  ( obj > this )  {^150^^^^^149^162^if  ( obj == this )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P15_Unwrap_Block]^return true;^150^151^152^^^149^162^if  ( obj == this )  { return true; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P16_Remove_Block]^^150^151^152^^^149^162^if  ( obj == this )  { return true; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P3_Replace_Literal]^return false;^151^^^^^149^162^return true;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P2_Replace_Operator]^if  ( ! ( obj  &  StandardXYSeriesLabelGenerator )  )  {^153^^^^^149^162^if  ( ! ( obj instanceof StandardXYSeriesLabelGenerator )  )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P15_Unwrap_Block]^return false;^153^154^155^^^149^162^if  ( ! ( obj instanceof StandardXYSeriesLabelGenerator )  )  { return false; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P16_Remove_Block]^^153^154^155^^^149^162^if  ( ! ( obj instanceof StandardXYSeriesLabelGenerator )  )  { return false; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P3_Replace_Literal]^return true;^154^^^^^149^162^return false;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P5_Replace_Variable]^if  ( !this.formatPattern.equals ( formatPattern )  )  {^158^^^^^149^162^if  ( !this.formatPattern.equals ( that.formatPattern )  )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P8_Replace_Mix]^if  ( !this.formatPattern.equals ( label )  )  {^158^^^^^149^162^if  ( !this.formatPattern.equals ( that.formatPattern )  )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P15_Unwrap_Block]^return false;^158^159^160^^^149^162^if  ( !this.formatPattern.equals ( that.formatPattern )  )  { return false; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P16_Remove_Block]^^158^159^160^^^149^162^if  ( !this.formatPattern.equals ( that.formatPattern )  )  { return false; }^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P3_Replace_Literal]^return true;^159^^^^^149^162^return false;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P5_Replace_Variable]^if  ( !this.formatPattern.equals ( that.formatPattern.formatPattern )  )  {^158^^^^^149^162^if  ( !this.formatPattern.equals ( that.formatPattern )  )  {^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P13_Insert_Block]^if  ( ! ( equals ( that.formatPattern )  )  )  {     return false; }^158^^^^^149^162^[Delete]^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
[P3_Replace_Literal]^return false;^161^^^^^149^162^return true;^[CLASS] StandardXYSeriesLabelGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  String  DEFAULT_LABEL_FORMAT  format  formatPattern  label  boolean  long  serialVersionUID  StandardXYSeriesLabelGenerator  that  
