[P1_Replace_Type]^private static final  short  serialVersionUID = 4019938025047800455L;^34^^^^^29^39^private static final long serialVersionUID = 4019938025047800455L;^[CLASS] DefaultTransformer   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID ;^34^^^^^29^39^private static final long serialVersionUID = 4019938025047800455L;^[CLASS] DefaultTransformer   [VARIABLES] 
[P2_Replace_Operator]^if  ( o != null )  {^45^^^^^43^58^if  ( o == null )  {^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P8_Replace_Mix]^if  ( o == false )  {^45^^^^^43^58^if  ( o == null )  {^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P15_Unwrap_Block]^throw new org.apache.commons.math.MathException("Conversion Exception in Transformation, Object is null");^45^46^47^^^43^58^if  ( o == null )  { throw new MathException  (" ")  ; }^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P16_Remove_Block]^^45^46^47^^^43^58^if  ( o == null )  { throw new MathException  (" ")  ; }^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P4_Replace_Constructor]^throw throw  new MathException (  ( "Conversion Exception in Transformation: " +  ( e.getMessage (  )  )  ) , e )   ;^46^^^^^43^58^throw new MathException  (" ")  ;^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P13_Insert_Block]^if  ( o == null )  {     throw new MathException ( "Conversion Exception in Transformation, Object is null" ) ; }^46^^^^^43^58^[Delete]^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P2_Replace_Operator]^if  ( o  ||  Number )  {^49^^^^^43^58^if  ( o instanceof Number )  {^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P15_Unwrap_Block]^return ((java.lang.Number) (o)).doubleValue();^49^50^51^^^43^58^if  ( o instanceof Number )  { return  (  ( Number ) o ) .doubleValue (  ) ; }^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P16_Remove_Block]^^49^50^51^^^43^58^if  ( o instanceof Number )  { return  (  ( Number ) o ) .doubleValue (  ) ; }^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P13_Insert_Block]^if  ( o instanceof Number )  {     return  (  ( Number )   ( o )  ) .doubleValue (  ) ; }^50^^^^^43^58^[Delete]^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P14_Delete_Statement]^^50^^^^^43^58^return  (  ( Number ) o ) .doubleValue (  ) ;^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P7_Replace_Invocation]^return new Double ( o.toString (  )  ) .Double (  ) ;^54^^^^^43^58^return new Double ( o.toString (  )  ) .doubleValue (  ) ;^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P7_Replace_Invocation]^return new Double ( o .Object (  )   ) .doubleValue (  ) ;^54^^^^^43^58^return new Double ( o.toString (  )  ) .doubleValue (  ) ;^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P14_Delete_Statement]^^54^^^^^43^58^return new Double ( o.toString (  )  ) .doubleValue (  ) ;^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P4_Replace_Constructor]^throw throw  new MathException ( "Conversion Exception in Transformation, Object is null" )   ;^56^^^^^43^58^throw new MathException  (" ")  ;^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
[P14_Delete_Statement]^^56^^^^^43^58^throw new MathException  (" ")  ;^[CLASS] DefaultTransformer  [METHOD] transform [RETURN_TYPE] double   Object o [VARIABLES] Object  o  boolean  long  serialVersionUID  Exception  e  
