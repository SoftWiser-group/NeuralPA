[P1_Replace_Type]^private static final  int  serialVersionUID = -2124469847758108312L;^33^^^^^28^38^private static final long serialVersionUID = -2124469847758108312L;^[CLASS] SortOrder   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID  = null ;^33^^^^^28^38^private static final long serialVersionUID = -2124469847758108312L;^[CLASS] SortOrder   [VARIABLES] 
[P3_Replace_Literal]^public static final SortOrder ASCENDING = new SortOrder ( "SortOortOrder.ASCENDING" ) ;^36^37^^^^36^37^public static final SortOrder ASCENDING = new SortOrder ( "SortOrder.ASCENDING" ) ;^[CLASS] SortOrder   [VARIABLES] 
[P3_Replace_Literal]^public static final SortOrder DESCENDING = new SortOrder ( "dortOrder.DESCENDING" ) ;^40^41^^^^40^41^public static final SortOrder DESCENDING = new SortOrder ( "SortOrder.DESCENDING" ) ;^[CLASS] SortOrder   [VARIABLES] 
[P1_Replace_Type]^private char name;^44^^^^^39^49^private String name;^[CLASS] SortOrder   [VARIABLES] 
[P8_Replace_Mix]^this.name =  null;^52^^^^^51^53^this.name = name;^[CLASS] SortOrder  [METHOD] <init> [RETURN_TYPE] String)   String name [VARIABLES] SortOrder  ASCENDING  DESCENDING  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^return name;^61^^^^^60^62^return this.name;^[CLASS] SortOrder  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] SortOrder  ASCENDING  DESCENDING  String  name  boolean  long  serialVersionUID  
[P2_Replace_Operator]^if  ( this <= obj )  {^74^^^^^72^87^if  ( this == obj )  {^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return true;^74^75^76^^^72^87^if  ( this == obj )  { return true; }^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^74^75^76^^^72^87^if  ( this == obj )  { return true; }^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return false;^75^^^^^72^87^return true;^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P2_Replace_Operator]^if  ( ! ( obj  ||  SortOrder )  )  {^77^^^^^72^87^if  ( ! ( obj instanceof SortOrder )  )  {^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return false;^77^78^79^^^72^87^if  ( ! ( obj instanceof SortOrder )  )  { return false; }^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^77^78^79^^^72^87^if  ( ! ( obj instanceof SortOrder )  )  { return false; }^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( equals ( that.toString (  )  )  )  )  {     return false; }^77^^^^^72^87^[Delete]^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return true;^78^^^^^72^87^return false;^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( !this.name.equals ( DESCENDING.toString (  )  )  )  {^82^^^^^72^87^if  ( !this.name.equals ( that.toString (  )  )  )  {^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^if  ( !this.name.equals ( that.SortOrder (  )  )  )  {^82^^^^^72^87^if  ( !this.name.equals ( that.toString (  )  )  )  {^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return false;^82^83^84^^^72^87^if  ( !this.name.equals ( that.toString (  )  )  )  { return false; }^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^82^83^84^^^72^87^if  ( !this.name.equals ( that.toString (  )  )  )  { return false; }^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( obj instanceof SortOrder )  )  {     return false; }^82^^^^^72^87^[Delete]^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return true;^83^^^^^72^87^return false;^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( equals ( that.toString (  )  )  )  )  {     return false; }^82^^^^^72^87^[Delete]^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^if  ( !this.name.equals ( that .SortOrder ( name )   )  )  {^82^^^^^72^87^if  ( !this.name.equals ( that.toString (  )  )  )  {^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return false;^86^^^^^72^87^return true;^[CLASS] SortOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  Object  obj  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^return name.hashCode (  ) ;^95^^^^^94^96^return this.name.hashCode (  ) ;^[CLASS] SortOrder  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^return this.name .hashCode (  )  ;^95^^^^^94^96^return this.name.hashCode (  ) ;^[CLASS] SortOrder  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P14_Delete_Statement]^^95^^^^^94^96^return this.name.hashCode (  ) ;^[CLASS] SortOrder  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( this.equals ( SortOrder.that )  )  {^106^^^^^105^113^if  ( this.equals ( SortOrder.ASCENDING )  )  {^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^if  ( this.toString ( SortOrder.ASCENDING )  )  {^106^^^^^105^113^if  ( this.equals ( SortOrder.ASCENDING )  )  {^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return org.jfree.chart.util.SortOrder.ASCENDING;^106^107^108^^^105^113^if  ( this.equals ( SortOrder.ASCENDING )  )  { return SortOrder.ASCENDING; }^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^106^107^108^^^105^113^if  ( this.equals ( SortOrder.ASCENDING )  )  { return SortOrder.ASCENDING; }^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^else if  ( this.equals ( SortOrder.that )  )  {^109^^^^^105^113^else if  ( this.equals ( SortOrder.DESCENDING )  )  {^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^else if  ( this.toString ( SortOrder.DESCENDING )  )  {^109^^^^^105^113^else if  ( this.equals ( SortOrder.DESCENDING )  )  {^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^else {^109^^^^^105^113^else if  ( this.equals ( SortOrder.DESCENDING )  )  {^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return org.jfree.chart.util.SortOrder.DESCENDING;^109^110^111^^^105^113^else if  ( this.equals ( SortOrder.DESCENDING )  )  { return SortOrder.DESCENDING; }^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^109^110^111^^^105^113^else if  ( this.equals ( SortOrder.DESCENDING )  )  { return SortOrder.DESCENDING; }^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^else if  ( this .equals ( 3 )   )  {^109^^^^^105^113^else if  ( this.equals ( SortOrder.DESCENDING )  )  {^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^else if  ( this.toString ( SortOrder.that )  )  {^109^^^^^105^113^else if  ( this.equals ( SortOrder.DESCENDING )  )  {^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^return false;^112^^^^^105^113^return null;^[CLASS] SortOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] SortOrder  ASCENDING  DESCENDING  that  String  name  boolean  long  serialVersionUID  
