[BugLab_Argument_Swapping]^if  ( y == x )  {^49^^^^^48^58^if  ( x == y )  {^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Wrong_Operator]^if  ( x != y )  {^49^^^^^48^58^if  ( x == y )  {^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Wrong_Literal]^return false;^50^^^^^48^58^return true;^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Argument_Swapping]^if  ( y == null || x == null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Wrong_Operator]^if  ( x == null && y == null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Wrong_Operator]^if  ( x != null || y == null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Wrong_Operator]^if  ( x == null || y != null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Wrong_Literal]^return true;^53^^^^^48^58^return false;^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Argument_Swapping]^return dty.equals ( dtx ) ;^57^^^^^48^58^return dtx.equals ( dty ) ;^[CLASS] PersistentLocalDateTime  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalDateTime  INSTANCE  LocalDateTime  dtx  dty  int[]  SQL_TYPES  Object  x  y  boolean  
[BugLab_Argument_Swapping]^return nullSafeGet ( strings, resultSet[0] ) ;^65^^^^^64^66^return nullSafeGet ( resultSet, strings[0] ) ;^[CLASS] PersistentLocalDateTime  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String[] strings Object object [VARIABLES] PersistentLocalDateTime  INSTANCE  ResultSet  resultSet  int[]  SQL_TYPES  Object  object  String[]  strings  boolean  
[BugLab_Wrong_Literal]^return nullSafeGet ( resultSet, strings[1] ) ;^65^^^^^64^66^return nullSafeGet ( resultSet, strings[0] ) ;^[CLASS] PersistentLocalDateTime  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String[] strings Object object [VARIABLES] PersistentLocalDateTime  INSTANCE  ResultSet  resultSet  int[]  SQL_TYPES  Object  object  String[]  strings  boolean  
[BugLab_Argument_Swapping]^Object timestamp = Hibernate.TIMESTAMP.nullSafeGet ( string, resultSet ) ;^69^^^^^68^74^Object timestamp = Hibernate.TIMESTAMP.nullSafeGet ( resultSet, string ) ;^[CLASS] PersistentLocalDateTime  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String string [VARIABLES] PersistentLocalDateTime  INSTANCE  ResultSet  resultSet  int[]  SQL_TYPES  Object  timestamp  String  string  boolean  
[BugLab_Wrong_Operator]^if  ( timestamp != null )  {^70^^^^^68^74^if  ( timestamp == null )  {^[CLASS] PersistentLocalDateTime  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String string [VARIABLES] PersistentLocalDateTime  INSTANCE  ResultSet  resultSet  int[]  SQL_TYPES  Object  timestamp  String  string  boolean  
[BugLab_Wrong_Operator]^if  ( value != null )  {^77^^^^^76^82^if  ( value == null )  {^[CLASS] PersistentLocalDateTime  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] PersistentLocalDateTime  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  boolean  int  index  
[BugLab_Argument_Swapping]^Hibernate.TIMESTAMP.nullSafeSet ( index,  (  ( LocalDateTime )  value ) .toDateTime (  ) .toDate (  ) , preparedStatement ) ;^80^^^^^76^82^Hibernate.TIMESTAMP.nullSafeSet ( preparedStatement,  (  ( LocalDateTime )  value ) .toDateTime (  ) .toDate (  ) , index ) ;^[CLASS] PersistentLocalDateTime  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] PersistentLocalDateTime  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  boolean  int  index  
[BugLab_Argument_Swapping]^Hibernate.TIMESTAMP.nullSafeSet ( index, null, preparedStatement ) ;^78^^^^^76^82^Hibernate.TIMESTAMP.nullSafeSet ( preparedStatement, null, index ) ;^[CLASS] PersistentLocalDateTime  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] PersistentLocalDateTime  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  boolean  int  index  
[BugLab_Wrong_Literal]^return true;^89^^^^^88^90^return false;^[CLASS] PersistentLocalDateTime  [METHOD] isMutable [RETURN_TYPE] boolean   [VARIABLES] PersistentLocalDateTime  INSTANCE  int[]  SQL_TYPES  boolean  
[BugLab_Variable_Misuse]^return target;^101^^^^^100^102^return original;^[CLASS] PersistentLocalDateTime  [METHOD] replace [RETURN_TYPE] Object   Object original Object target Object owner [VARIABLES] PersistentLocalDateTime  INSTANCE  int[]  SQL_TYPES  Object  original  owner  target  boolean  
