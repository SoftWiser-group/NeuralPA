[P3_Replace_Literal]^private static final String[] STRING_ARRAY = new String[-1];^33^^^^^28^38^private static final String[] STRING_ARRAY = new String[0];^[CLASS] PageScopeContextHandler   [VARIABLES] 
[P8_Replace_Mix]^private static final String[] STRING_ARRAY = new String[0 >>> 4];^33^^^^^28^38^private static final String[] STRING_ARRAY = new String[0];^[CLASS] PageScopeContextHandler   [VARIABLES] 
[P7_Replace_Invocation]^Enumeration e =  (  ( PageScopeContext )  pageScope ) .getAttribute (  ) ;^36^^^^^35^42^Enumeration e =  (  ( PageScopeContext )  pageScope ) .getAttributeNames (  ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P14_Delete_Statement]^^36^^^^^35^42^Enumeration e =  (  ( PageScopeContext )  pageScope ) .getAttributeNames (  ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P11_Insert_Donor_Statement]^return  (  ( PageScopeContext )  pageScope ) .getAttribute ( property ) ;Enumeration e =  (  ( PageScopeContext )  pageScope ) .getAttributeNames (  ) ;^36^^^^^35^42^Enumeration e =  (  ( PageScopeContext )  pageScope ) .getAttributeNames (  ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P11_Insert_Donor_Statement]^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;Enumeration e =  (  ( PageScopeContext )  pageScope ) .getAttributeNames (  ) ;^36^^^^^35^42^Enumeration e =  (  ( PageScopeContext )  pageScope ) .getAttributeNames (  ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P1_Replace_Type]^LinkedList  list = new  LinkedList  ( 16 ) ;^37^^^^^35^42^ArrayList list = new ArrayList ( 16 ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P3_Replace_Literal]^ArrayList list = new ArrayList ( this ) ;^37^^^^^35^42^ArrayList list = new ArrayList ( 16 ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P3_Replace_Literal]^ArrayList list = new ArrayList ( 9 ) ;^37^^^^^35^42^ArrayList list = new ArrayList ( 16 ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P8_Replace_Mix]^while  ( e .nextElement (  )   )  {^38^^^^^35^42^while  ( e.hasMoreElements (  )  )  {^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P14_Delete_Statement]^^39^^^^^35^42^list.add ( e.nextElement (  )  ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P7_Replace_Invocation]^list.add ( e .hasMoreElements (  )   ) ;^39^^^^^35^42^list.add ( e.nextElement (  )  ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P14_Delete_Statement]^^38^39^^^^35^42^while  ( e.hasMoreElements (  )  )  { list.add ( e.nextElement (  )  ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P5_Replace_Variable]^return  ( String[] )  STRING_ARRAY.toArray ( list ) ;^41^^^^^35^42^return  ( String[] )  list.toArray ( STRING_ARRAY ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P7_Replace_Invocation]^return  ( String[] )  list.ArrayList ( STRING_ARRAY ) ;^41^^^^^35^42^return  ( String[] )  list.toArray ( STRING_ARRAY ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P14_Delete_Statement]^^41^^^^^35^42^return  ( String[] )  list.toArray ( STRING_ARRAY ) ;^[CLASS] PageScopeContextHandler  [METHOD] getPropertyNames [RETURN_TYPE] String[]   Object pageScope [VARIABLES] ArrayList  list  Object  pageScope  String[]  STRING_ARRAY  Enumeration  e  boolean  
[P7_Replace_Invocation]^return  (  ( PageScopeContext )  pageScope ) .setAttribute ( property ) ;^45^^^^^44^46^return  (  ( PageScopeContext )  pageScope ) .getAttribute ( property ) ;^[CLASS] PageScopeContextHandler  [METHOD] getProperty [RETURN_TYPE] Object   Object pageScope String property [VARIABLES] Object  pageScope  String[]  STRING_ARRAY  String  property  boolean  
[P14_Delete_Statement]^^45^^^^^44^46^return  (  ( PageScopeContext )  pageScope ) .getAttribute ( property ) ;^[CLASS] PageScopeContextHandler  [METHOD] getProperty [RETURN_TYPE] Object   Object pageScope String property [VARIABLES] Object  pageScope  String[]  STRING_ARRAY  String  property  boolean  
[P5_Replace_Variable]^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, pageScope ) ;^49^^^^^48^50^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^[CLASS] PageScopeContextHandler  [METHOD] setProperty [RETURN_TYPE] void   Object pageScope String property Object value [VARIABLES] Object  pageScope  value  String[]  STRING_ARRAY  String  property  boolean  
[P5_Replace_Variable]^(  ( PageScopeContext )  pageScope ) .setAttribute (  value ) ;^49^^^^^48^50^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^[CLASS] PageScopeContextHandler  [METHOD] setProperty [RETURN_TYPE] void   Object pageScope String property Object value [VARIABLES] Object  pageScope  value  String[]  STRING_ARRAY  String  property  boolean  
[P5_Replace_Variable]^(  ( PageScopeContext )  pageScope ) .setAttribute ( property ) ;^49^^^^^48^50^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^[CLASS] PageScopeContextHandler  [METHOD] setProperty [RETURN_TYPE] void   Object pageScope String property Object value [VARIABLES] Object  pageScope  value  String[]  STRING_ARRAY  String  property  boolean  
[P5_Replace_Variable]^(  ( PageScopeContext )  pageScope ) .setAttribute ( value, property ) ;^49^^^^^48^50^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^[CLASS] PageScopeContextHandler  [METHOD] setProperty [RETURN_TYPE] void   Object pageScope String property Object value [VARIABLES] Object  pageScope  value  String[]  STRING_ARRAY  String  property  boolean  
[P7_Replace_Invocation]^(  ( PageScopeContext )  pageScope )  .getAttribute ( property )  ;^49^^^^^48^50^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^[CLASS] PageScopeContextHandler  [METHOD] setProperty [RETURN_TYPE] void   Object pageScope String property Object value [VARIABLES] Object  pageScope  value  String[]  STRING_ARRAY  String  property  boolean  
[P14_Delete_Statement]^^49^^^^^48^50^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^[CLASS] PageScopeContextHandler  [METHOD] setProperty [RETURN_TYPE] void   Object pageScope String property Object value [VARIABLES] Object  pageScope  value  String[]  STRING_ARRAY  String  property  boolean  
[P11_Insert_Donor_Statement]^Enumeration e =  (  ( PageScopeContext )  pageScope ) .getAttributeNames (  ) ;(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^49^^^^^48^50^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^[CLASS] PageScopeContextHandler  [METHOD] setProperty [RETURN_TYPE] void   Object pageScope String property Object value [VARIABLES] Object  pageScope  value  String[]  STRING_ARRAY  String  property  boolean  
[P11_Insert_Donor_Statement]^return  (  ( PageScopeContext )  pageScope ) .getAttribute ( property ) ;(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^49^^^^^48^50^(  ( PageScopeContext )  pageScope ) .setAttribute ( property, value ) ;^[CLASS] PageScopeContextHandler  [METHOD] setProperty [RETURN_TYPE] void   Object pageScope String property Object value [VARIABLES] Object  pageScope  value  String[]  STRING_ARRAY  String  property  boolean  
