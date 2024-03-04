[BugLab_Wrong_Literal]^private static final int STREAM_BUFFER_LENGTH = 1025;^35^^^^^30^40^private static final int STREAM_BUFFER_LENGTH = 1024;^[CLASS] DigestUtils   [VARIABLES] 
[BugLab_Argument_Swapping]^int read = buffer.read ( data, 0, STREAM_BUFFER_LENGTH ) ;^132^^^^^130^140^int read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Argument_Swapping]^int read = data.read ( STREAM_BUFFER_LENGTH, 0, buffer ) ;^132^^^^^130^140^int read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Argument_Swapping]^int read = STREAM_BUFFER_LENGTH.read ( buffer, 0, data ) ;^132^^^^^130^140^int read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Wrong_Literal]^int read = data.read ( buffer, , STREAM_BUFFER_LENGTH ) ;^132^^^^^130^140^int read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Variable_Misuse]^int read = data.read ( buffer, 0, read ) ;^132^^^^^130^140^int read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Wrong_Literal]^int read = data.read ( buffer, STREAM_BUFFER_LENGTH, STREAM_BUFFER_LENGTH ) ;^132^^^^^130^140^int read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Wrong_Operator]^while ( read >= -1 )  {^134^^^^^130^140^while ( read > -1 )  {^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Wrong_Literal]^while ( read > -read )  {^134^^^^^130^140^while ( read > -1 )  {^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Variable_Misuse]^read = data.read ( buffer, 0, read ) ;^136^^^^^130^140^read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Argument_Swapping]^read = STREAM_BUFFER_LENGTH.read ( buffer, 0, data ) ;^136^^^^^130^140^read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Argument_Swapping]^read = buffer.read ( data, 0, STREAM_BUFFER_LENGTH ) ;^136^^^^^130^140^read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Argument_Swapping]^read = data.read ( STREAM_BUFFER_LENGTH, 0, buffer ) ;^136^^^^^130^140^read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Wrong_Literal]^read = data.read ( buffer, read, STREAM_BUFFER_LENGTH ) ;^136^^^^^130^140^read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Variable_Misuse]^digest.update ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^135^^^^^130^140^digest.update ( buffer, 0, read ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Argument_Swapping]^digest.update ( read, 0, buffer ) ;^135^^^^^130^140^digest.update ( buffer, 0, read ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
[BugLab_Wrong_Literal]^read = data.read ( buffer, 1, STREAM_BUFFER_LENGTH ) ;^136^^^^^130^140^read = data.read ( buffer, 0, STREAM_BUFFER_LENGTH ) ;^[CLASS] DigestUtils  [METHOD] digest [RETURN_TYPE] byte[]   MessageDigest digest InputStream data [VARIABLES] byte[]  buffer  boolean  MessageDigest  digest  int  STREAM_BUFFER_LENGTH  read  InputStream  data  
