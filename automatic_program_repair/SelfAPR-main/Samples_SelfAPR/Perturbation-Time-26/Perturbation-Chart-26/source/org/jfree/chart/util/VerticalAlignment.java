[P1_Replace_Type]^private static final  short  serialVersionUID = 7272397034325429853L;^56^^^^^51^61^private static final long serialVersionUID = 7272397034325429853L;^[CLASS] VerticalAlignment   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID  = null ;^56^^^^^51^61^private static final long serialVersionUID = 7272397034325429853L;^[CLASS] VerticalAlignment   [VARIABLES] 
[P3_Replace_Literal]^public static final VerticalAlignment TOP = new VerticalAlignment ( "VerticalAlignment.TOPVe" ) ;^59^60^^^^59^60^public static final VerticalAlignment TOP = new VerticalAlignment ( "VerticalAlignment.TOP" ) ;^[CLASS] VerticalAlignment   [VARIABLES] 
[P3_Replace_Literal]^public static final VerticalAlignment BOTTOM = new VerticalAlignment ( "Vertica" ) ;^63^64^^^^63^64^public static final VerticalAlignment BOTTOM = new VerticalAlignment ( "VerticalAlignment.BOTTOM" ) ;^[CLASS] VerticalAlignment   [VARIABLES] 
[P3_Replace_Literal]^public static final VerticalAlignment CENTER = new VerticalAlignment ( "alAerticalAlignment.CENTER" ) ;^67^68^^^^67^68^public static final VerticalAlignment CENTER = new VerticalAlignment ( "VerticalAlignment.CENTER" ) ;^[CLASS] VerticalAlignment   [VARIABLES] 
[P1_Replace_Type]^private char name;^71^^^^^66^76^private String name;^[CLASS] VerticalAlignment   [VARIABLES] 
[P8_Replace_Mix]^this.name =  null;^79^^^^^78^80^this.name = name;^[CLASS] VerticalAlignment  [METHOD] <init> [RETURN_TYPE] String)   String name [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^return name;^88^^^^^87^89^return this.name;^[CLASS] VerticalAlignment  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  String  name  boolean  long  serialVersionUID  
[P2_Replace_Operator]^if  ( this >= obj )  {^100^^^^^99^112^if  ( this == obj )  {^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return true;^100^101^102^^^99^112^if  ( this == obj )  { return true; }^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^100^101^102^^^99^112^if  ( this == obj )  { return true; }^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return false;^101^^^^^99^112^return true;^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P2_Replace_Operator]^if  ( ! ( obj  &  VerticalAlignment )  )  {^103^^^^^99^112^if  ( ! ( obj instanceof VerticalAlignment )  )  {^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^if  (  ( obj instanceof VerticalAlignment )  )  {^103^^^^^99^112^if  ( ! ( obj instanceof VerticalAlignment )  )  {^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return false;^103^104^105^^^99^112^if  ( ! ( obj instanceof VerticalAlignment )  )  { return false; }^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^103^104^105^^^99^112^if  ( ! ( obj instanceof VerticalAlignment )  )  { return false; }^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( equals ( alignment.name )  )  )  {     return false; }^103^^^^^99^112^[Delete]^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return true;^104^^^^^99^112^return false;^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( !this.name.equals ( TOP.name )  )  {^108^^^^^99^112^if  ( !this.name.equals ( alignment.name )  )  {^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^if  ( !this.name .VerticalAlignment ( name )   )  {^108^^^^^99^112^if  ( !this.name.equals ( alignment.name )  )  {^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return false;^108^109^110^^^99^112^if  ( !this.name.equals ( alignment.name )  )  { return false; }^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^108^109^110^^^99^112^if  ( !this.name.equals ( alignment.name )  )  { return false; }^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( obj instanceof VerticalAlignment )  )  {     return false; }^108^^^^^99^112^[Delete]^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return true;^109^^^^^99^112^return false;^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( !this.name.equals ( name )  )  {^108^^^^^99^112^if  ( !this.name.equals ( alignment.name )  )  {^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( !this.name.equals ( alignment.name.name )  )  {^108^^^^^99^112^if  ( !this.name.equals ( alignment.name )  )  {^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( !this.name.equals ( alignment )  )  {^108^^^^^99^112^if  ( !this.name.equals ( alignment.name )  )  {^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( equals ( alignment.name )  )  )  {     return false; }^108^^^^^99^112^[Delete]^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P3_Replace_Literal]^return false;^111^^^^^99^112^return true;^[CLASS] VerticalAlignment  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  Object  obj  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^return name.hashCode (  ) ;^120^^^^^119^121^return this.name.hashCode (  ) ;^[CLASS] VerticalAlignment  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^return this.name .hashCode (  )  ;^120^^^^^119^121^return this.name.hashCode (  ) ;^[CLASS] VerticalAlignment  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P14_Delete_Statement]^^120^^^^^119^121^return this.name.hashCode (  ) ;^[CLASS] VerticalAlignment  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^if  ( this.equals ( VerticalAlignment.alignment )  )  {^131^^^^^130^143^if  ( this.equals ( VerticalAlignment.TOP )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return org.jfree.chart.util.VerticalAlignment.TOP;^131^132^133^^^130^143^if  ( this.equals ( VerticalAlignment.TOP )  )  { return VerticalAlignment.TOP; }^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^131^132^133^^^130^143^if  ( this.equals ( VerticalAlignment.TOP )  )  { return VerticalAlignment.TOP; }^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( this.equals ( BOTTOM )  )  {     return BOTTOM; }else     if  ( this.equals ( CENTER )  )  {         return CENTER;     }else {         return null;     }^131^^^^^130^143^[Delete]^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^else if  ( this.equals ( VerticalAlignment.alignment )  )  {^134^^^^^130^143^else if  ( this.equals ( VerticalAlignment.BOTTOM )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^else if  ( this .VerticalAlignment ( name )   )  {^134^^^^^130^143^else if  ( this.equals ( VerticalAlignment.BOTTOM )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^else {^134^^^^^130^143^else if  ( this.equals ( VerticalAlignment.BOTTOM )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return org.jfree.chart.util.VerticalAlignment.BOTTOM;^134^135^136^^^130^143^else if  ( this.equals ( VerticalAlignment.BOTTOM )  )  { return VerticalAlignment.BOTTOM; }^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^134^135^136^^^130^143^else if  ( this.equals ( VerticalAlignment.BOTTOM )  )  { return VerticalAlignment.BOTTOM; }^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( this.equals ( TOP )  )  {     return TOP; }else     if  ( this.equals ( BOTTOM )  )  {         return BOTTOM;     }else         if  ( this.equals ( CENTER )  )  {             return CENTER;         }else {             return null;         }^134^^^^^130^143^[Delete]^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P5_Replace_Variable]^else if  ( this.equals ( VerticalAlignment.alignment )  )  {^137^^^^^130^143^else if  ( this.equals ( VerticalAlignment.CENTER )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^if  ( this.equals ( VerticalAlignment.alignment )  )  {^137^^^^^130^143^else if  ( this.equals ( VerticalAlignment.CENTER )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P9_Replace_Statement]^else if  ( this.equals ( VerticalAlignment.BOTTOM )  )  {^137^^^^^130^143^else if  ( this.equals ( VerticalAlignment.CENTER )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P15_Unwrap_Block]^return org.jfree.chart.util.VerticalAlignment.CENTER;^137^138^139^^^130^143^else if  ( this.equals ( VerticalAlignment.CENTER )  )  { return VerticalAlignment.CENTER; }^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P16_Remove_Block]^^137^138^139^^^130^143^else if  ( this.equals ( VerticalAlignment.CENTER )  )  { return VerticalAlignment.CENTER; }^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^return this;^141^^^^^130^143^return null;^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( obj instanceof VerticalAlignment )  )  {     return false; }^138^^^^^130^143^[Delete]^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P13_Insert_Block]^if  ( ! ( obj instanceof VerticalAlignment )  )  {     return false; }^137^^^^^130^143^[Delete]^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^else if  ( this .VerticalAlignment ( name )   )  {^137^^^^^130^143^else if  ( this.equals ( VerticalAlignment.CENTER )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^return false;^141^^^^^130^143^return null;^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^else {^137^^^^^130^143^else if  ( this.equals ( VerticalAlignment.CENTER )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P8_Replace_Mix]^return true;^141^^^^^130^143^return null;^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
[P7_Replace_Invocation]^if  ( this .VerticalAlignment ( name )   )  {^131^^^^^130^143^if  ( this.equals ( VerticalAlignment.TOP )  )  {^[CLASS] VerticalAlignment  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] VerticalAlignment  BOTTOM  CENTER  TOP  alignment  String  name  boolean  long  serialVersionUID  
