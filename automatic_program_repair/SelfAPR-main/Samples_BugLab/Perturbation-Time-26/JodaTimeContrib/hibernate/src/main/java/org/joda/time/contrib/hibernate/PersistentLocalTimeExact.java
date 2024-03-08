[BugLab_Argument_Swapping]^if  ( y == x )  {^49^^^^^48^58^if  ( x == y )  {^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Wrong_Operator]^if  ( x != y )  {^49^^^^^48^58^if  ( x == y )  {^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Wrong_Literal]^return false;^50^^^^^48^58^return true;^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Argument_Swapping]^if  ( y == null || x == null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Wrong_Operator]^if  ( x == null && y == null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Wrong_Operator]^if  ( x != null || y == null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Wrong_Operator]^if  ( x == null || y != null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Wrong_Literal]^return true;^53^^^^^48^58^return false;^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Argument_Swapping]^return dty.equals ( dtx ) ;^57^^^^^48^58^return dtx.equals ( dty ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[BugLab_Argument_Swapping]^return nullSafeGet ( strings, resultSet[0] ) ;^65^^^^^64^67^return nullSafeGet ( resultSet, strings[0] ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String[] strings Object object [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  object  String[]  strings  boolean  
[BugLab_Wrong_Literal]^return nullSafeGet ( resultSet, strings[-1] ) ;^65^^^^^64^67^return nullSafeGet ( resultSet, strings[0] ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String[] strings Object object [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  object  String[]  strings  boolean  
[BugLab_Argument_Swapping]^Object timestamp = Hibernate.INTEGER.nullSafeGet ( string, resultSet ) ;^70^^^^^69^76^Object timestamp = Hibernate.INTEGER.nullSafeGet ( resultSet, string ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String string [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  timestamp  String  string  boolean  
[BugLab_Wrong_Operator]^if  ( timestamp != null )  {^71^^^^^69^76^if  ( timestamp == null )  {^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String string [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  timestamp  String  string  boolean  
[BugLab_Wrong_Operator]^if  ( value != null )  {^79^^^^^78^85^if  ( value == null )  {^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[BugLab_Argument_Swapping]^Hibernate.INTEGER.nullSafeSet ( lt, new Integer ( preparedStatement.getMillisOfDay (  )  ) , index ) ;^83^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[BugLab_Argument_Swapping]^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( index.getMillisOfDay (  )  ) , lt ) ;^83^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[BugLab_Argument_Swapping]^Hibernate.INTEGER.nullSafeSet ( index, null, preparedStatement ) ;^80^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, null, index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[BugLab_Argument_Swapping]^Hibernate.INTEGER.nullSafeSet ( index, new Integer ( lt.getMillisOfDay (  )  ) , preparedStatement ) ;^83^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[BugLab_Wrong_Literal]^return true;^92^^^^^91^93^return false;^[CLASS] PersistentLocalTimeExact  [METHOD] isMutable [RETURN_TYPE] boolean   [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  boolean  
[BugLab_Variable_Misuse]^return target;^104^^^^^103^105^return original;^[CLASS] PersistentLocalTimeExact  [METHOD] replace [RETURN_TYPE] Object   Object original Object target Object owner [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  original  owner  target  boolean  