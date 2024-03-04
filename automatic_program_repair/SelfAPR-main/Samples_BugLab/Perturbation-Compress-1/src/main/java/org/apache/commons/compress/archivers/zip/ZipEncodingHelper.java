[BugLab_Variable_Misuse]^if  ( encoding == null )  {^58^^^^^57^62^if  ( this.encoding == null )  {^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getEncoding [RETURN_TYPE] Simple8BitZipEncoding   [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  char[]  highChars  
[BugLab_Wrong_Operator]^if  ( this.encoding != null )  {^58^^^^^57^62^if  ( this.encoding == null )  {^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getEncoding [RETURN_TYPE] Simple8BitZipEncoding   [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^this.encoding = new Simple8BitZipEncoding ( highChars ) ;^59^^^^^57^62^this.encoding = new Simple8BitZipEncoding ( this.highChars ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getEncoding [RETURN_TYPE] Simple8BitZipEncoding   [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^return encoding;^61^^^^^57^62^return this.encoding;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getEncoding [RETURN_TYPE] Simple8BitZipEncoding   [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^b.limit ( on.position (  )  ) ;^148^^^^^147^156^b.limit ( b.position (  )  ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] growBuffer [RETURN_TYPE] ByteBuffer   ByteBuffer b int newCapacity [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  b  on  int  c2  newCapacity  char[]  highChars  
[BugLab_Variable_Misuse]^int c2 = on.capacity (  )  * 2;^151^^^^^147^156^int c2 = b.capacity (  )  * 2;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] growBuffer [RETURN_TYPE] ByteBuffer   ByteBuffer b int newCapacity [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  b  on  int  c2  newCapacity  char[]  highChars  
[BugLab_Wrong_Operator]^int + c2 = b.capacity (  )  * 2;^151^^^^^147^156^int c2 = b.capacity (  )  * 2;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] growBuffer [RETURN_TYPE] ByteBuffer   ByteBuffer b int newCapacity [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  b  on  int  c2  newCapacity  char[]  highChars  
[BugLab_Wrong_Literal]^int cnewCapacity = b.capacity (  )  * newCapacity;^151^^^^^147^156^int c2 = b.capacity (  )  * 2;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] growBuffer [RETURN_TYPE] ByteBuffer   ByteBuffer b int newCapacity [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  b  on  int  c2  newCapacity  char[]  highChars  
[BugLab_Variable_Misuse]^ByteBuffer on = ByteBuffer.allocate ( newCapacity < newCapacity ? newCapacity : c2 ) ;^152^^^^^147^156^ByteBuffer on = ByteBuffer.allocate ( c2 < newCapacity ? newCapacity : c2 ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] growBuffer [RETURN_TYPE] ByteBuffer   ByteBuffer b int newCapacity [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  b  on  int  c2  newCapacity  char[]  highChars  
[BugLab_Argument_Swapping]^ByteBuffer on = ByteBuffer.allocate ( newCapacity < c2 ? newCapacity : c2 ) ;^152^^^^^147^156^ByteBuffer on = ByteBuffer.allocate ( c2 < newCapacity ? newCapacity : c2 ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] growBuffer [RETURN_TYPE] ByteBuffer   ByteBuffer b int newCapacity [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  b  on  int  c2  newCapacity  char[]  highChars  
[BugLab_Wrong_Operator]^ByteBuffer on = ByteBuffer.allocate ( c2 <= newCapacity ? newCapacity : c2 ) ;^152^^^^^147^156^ByteBuffer on = ByteBuffer.allocate ( c2 < newCapacity ? newCapacity : c2 ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] growBuffer [RETURN_TYPE] ByteBuffer   ByteBuffer b int newCapacity [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  b  on  int  c2  newCapacity  char[]  highChars  
[BugLab_Variable_Misuse]^return b;^155^^^^^147^156^return on;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] growBuffer [RETURN_TYPE] ByteBuffer   ByteBuffer b int newCapacity [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  b  on  int  c2  newCapacity  char[]  highChars  
[BugLab_Argument_Swapping]^bb.put ( c[ ( HEX_DIGITS >> 12 ) &0x0f] ) ;^181^^^^^176^185^bb.put ( HEX_DIGITS[ ( c >> 12 ) &0x0f] ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] appendSurrogate [RETURN_TYPE] void   ByteBuffer bb char c [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  char  c  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  bb  char[]  highChars  
[BugLab_Wrong_Operator]^bb.put ( HEX_DIGITS[ ( c  ==  12 ) &0x0f] ) ;^181^^^^^176^185^bb.put ( HEX_DIGITS[ ( c >> 12 ) &0x0f] ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] appendSurrogate [RETURN_TYPE] void   ByteBuffer bb char c [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  char  c  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  bb  char[]  highChars  
[BugLab_Wrong_Literal]^bb.put ( HEX_DIGITS[ ( c >>  ) &0x0f] ) ;^181^^^^^176^185^bb.put ( HEX_DIGITS[ ( c >> 12 ) &0x0f] ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] appendSurrogate [RETURN_TYPE] void   ByteBuffer bb char c [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  char  c  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  bb  char[]  highChars  
[BugLab_Wrong_Operator]^bb.put ( HEX_DIGITS[ ( c  >  8 ) &0x0f] ) ;^182^^^^^176^185^bb.put ( HEX_DIGITS[ ( c >> 8 ) &0x0f] ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] appendSurrogate [RETURN_TYPE] void   ByteBuffer bb char c [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  char  c  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  bb  char[]  highChars  
[BugLab_Wrong_Literal]^bb.put ( HEX_DIGITS[ ( c >> 7 ) &0x0f] ) ;^182^^^^^176^185^bb.put ( HEX_DIGITS[ ( c >> 8 ) &0x0f] ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] appendSurrogate [RETURN_TYPE] void   ByteBuffer bb char c [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  char  c  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  bb  char[]  highChars  
[BugLab_Argument_Swapping]^bb.put ( c[ ( HEX_DIGITS >> 4 ) &0x0f] ) ;^183^^^^^176^185^bb.put ( HEX_DIGITS[ ( c >> 4 ) &0x0f] ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] appendSurrogate [RETURN_TYPE] void   ByteBuffer bb char c [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  char  c  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  bb  char[]  highChars  
[BugLab_Wrong_Operator]^bb.put ( HEX_DIGITS[ ( c  >  4 ) &0x0f] ) ;^183^^^^^176^185^bb.put ( HEX_DIGITS[ ( c >> 4 ) &0x0f] ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] appendSurrogate [RETURN_TYPE] void   ByteBuffer bb char c [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  char  c  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  bb  char[]  highChars  
[BugLab_Wrong_Operator]^bb.put ( HEX_DIGITS[c  ==  0x0f] ) ;^184^^^^^176^185^bb.put ( HEX_DIGITS[c & 0x0f] ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] appendSurrogate [RETURN_TYPE] void   ByteBuffer bb char c [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  char  c  Simple8BitZipEncoding  encoding  String  UTF8  Map  simpleEncodings  ByteBuffer  bb  char[]  highChars  
[BugLab_Variable_Misuse]^if  ( isUTF8 ( UTF8 )  )  {^208^^^^^205^231^if  ( isUTF8 ( name )  )  {^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^if  ( UTF8 == null )  {^212^^^^^205^231^if  ( name == null )  {^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Wrong_Operator]^if  ( name != null )  {^212^^^^^205^231^if  ( name == null )  {^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^SimpleEncodingHolder h = ( SimpleEncodingHolder )  simpleEncodings.get ( UTF8 ) ;^216^217^^^^205^231^SimpleEncodingHolder h = ( SimpleEncodingHolder )  simpleEncodings.get ( name ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Argument_Swapping]^SimpleEncodingHolder h = ( SimpleEncodingHolder )  name.get ( simpleEncodings ) ;^216^217^^^^205^231^SimpleEncodingHolder h = ( SimpleEncodingHolder )  simpleEncodings.get ( name ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^( SimpleEncodingHolder )  simpleEncodings.get ( UTF8 ) ;^217^^^^^205^231^( SimpleEncodingHolder )  simpleEncodings.get ( name ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Argument_Swapping]^( SimpleEncodingHolder )  name.get ( simpleEncodings ) ;^217^^^^^205^231^( SimpleEncodingHolder )  simpleEncodings.get ( name ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^Charset cs = Charset.forName ( UTF8 ) ;^225^^^^^205^231^Charset cs = Charset.forName ( name ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^return new FallbackZipEncoding ( UTF8 ) ;^229^^^^^205^231^return new FallbackZipEncoding ( name ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] getZipEncoding [RETURN_TYPE] ZipEncoding   String name [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  SimpleEncodingHolder  h  UnsupportedCharsetException  e  Charset  cs  Simple8BitZipEncoding  encoding  String  UTF8  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Wrong_Operator]^if  ( encoding != null )  {^238^^^^^237^244^if  ( encoding == null )  {^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] isUTF8 [RETURN_TYPE] boolean   String encoding [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  encoding  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^return UTF8.equalsIgnoreCase ( name ) || "utf-8".equalsIgnoreCase ( encoding ) ;^242^243^^^^237^244^return UTF8.equalsIgnoreCase ( encoding ) || "utf-8".equalsIgnoreCase ( encoding ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] isUTF8 [RETURN_TYPE] boolean   String encoding [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  encoding  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^return name.equalsIgnoreCase ( encoding ) || "utf-8".equalsIgnoreCase ( encoding ) ;^242^243^^^^237^244^return UTF8.equalsIgnoreCase ( encoding ) || "utf-8".equalsIgnoreCase ( encoding ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] isUTF8 [RETURN_TYPE] boolean   String encoding [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  encoding  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Argument_Swapping]^return encoding.equalsIgnoreCase ( UTF8 ) || "utf-8".equalsIgnoreCase ( encoding ) ;^242^243^^^^237^244^return UTF8.equalsIgnoreCase ( encoding ) || "utf-8".equalsIgnoreCase ( encoding ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] isUTF8 [RETURN_TYPE] boolean   String encoding [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  encoding  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Wrong_Operator]^return UTF8.equalsIgnoreCase ( encoding ) && "utf-8".equalsIgnoreCase ( encoding ) ;^242^243^^^^237^244^return UTF8.equalsIgnoreCase ( encoding ) || "utf-8".equalsIgnoreCase ( encoding ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] isUTF8 [RETURN_TYPE] boolean   String encoding [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  encoding  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^|| "utf-8".equalsIgnoreCase ( name ) ;^243^^^^^237^244^|| "utf-8".equalsIgnoreCase ( encoding ) ;^[CLASS] ZipEncodingHelper SimpleEncodingHolder  [METHOD] isUTF8 [RETURN_TYPE] boolean   String encoding [VARIABLES] byte[]  HEX_DIGITS  boolean  ZipEncoding  UTF8_ZIP_ENCODING  Simple8BitZipEncoding  encoding  String  UTF8  encoding  name  Map  simpleEncodings  char[]  highChars  
[BugLab_Variable_Misuse]^if  ( encoding == null )  {^58^^^^^57^62^if  ( this.encoding == null )  {^[CLASS] SimpleEncodingHolder  [METHOD] getEncoding [RETURN_TYPE] Simple8BitZipEncoding   [VARIABLES] Simple8BitZipEncoding  encoding  char[]  highChars  boolean  
[BugLab_Wrong_Operator]^if  ( this.encoding != null )  {^58^^^^^57^62^if  ( this.encoding == null )  {^[CLASS] SimpleEncodingHolder  [METHOD] getEncoding [RETURN_TYPE] Simple8BitZipEncoding   [VARIABLES] Simple8BitZipEncoding  encoding  char[]  highChars  boolean  
[BugLab_Variable_Misuse]^this.encoding = new Simple8BitZipEncoding ( highChars ) ;^59^^^^^57^62^this.encoding = new Simple8BitZipEncoding ( this.highChars ) ;^[CLASS] SimpleEncodingHolder  [METHOD] getEncoding [RETURN_TYPE] Simple8BitZipEncoding   [VARIABLES] Simple8BitZipEncoding  encoding  char[]  highChars  boolean  
[BugLab_Variable_Misuse]^return encoding;^61^^^^^57^62^return this.encoding;^[CLASS] SimpleEncodingHolder  [METHOD] getEncoding [RETURN_TYPE] Simple8BitZipEncoding   [VARIABLES] Simple8BitZipEncoding  encoding  char[]  highChars  boolean  
