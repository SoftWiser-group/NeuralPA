[BugLab_Wrong_Literal]^private static final BitSet PRINTABLE_CHARS = new BitSet (  ) ;^57^^^^^52^62^private static final BitSet PRINTABLE_CHARS = new BitSet ( 256 ) ;^[CLASS] QCodec   [VARIABLES] 
[BugLab_Wrong_Literal]^private static final byte BLANK = 33;^102^^^^^97^107^private static final byte BLANK = 32;^[CLASS] QCodec   [VARIABLES] 
[BugLab_Wrong_Literal]^private static final byte UNDERSCORE = 96;^104^^^^^99^109^private static final byte UNDERSCORE = 95;^[CLASS] QCodec   [VARIABLES] 
[BugLab_Wrong_Literal]^private boolean encodeBlanks = true;^106^^^^^101^111^private boolean encodeBlanks = false;^[CLASS] QCodec   [VARIABLES] 
[BugLab_Variable_Misuse]^if  ( data == null )  {^134^^^^^133^146^if  ( bytes == null )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Wrong_Operator]^if  ( bytes != null )  {^134^^^^^133^146^if  ( bytes == null )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Argument_Swapping]^byte[] data = QuotedPrintableCodec.encodeQuotedPrintable ( bytes, PRINTABLE_CHARS ) ;^137^^^^^133^146^byte[] data = QuotedPrintableCodec.encodeQuotedPrintable ( PRINTABLE_CHARS, bytes ) ;^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Variable_Misuse]^byte[] data = QuotedPrintableCodec.encodeQuotedPrintable ( PRINTABLE_CHARS, data ) ;^137^^^^^133^146^byte[] data = QuotedPrintableCodec.encodeQuotedPrintable ( PRINTABLE_CHARS, bytes ) ;^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Variable_Misuse]^if  ( encodeBlanks )  {^138^^^^^133^146^if  ( this.encodeBlanks )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Variable_Misuse]^if  ( bytes[i] == BLANK )  {^140^^^^^133^146^if  ( data[i] == BLANK )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Argument_Swapping]^if  ( BLANK[i] == data )  {^140^^^^^133^146^if  ( data[i] == BLANK )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Wrong_Operator]^if  ( data[i] != BLANK )  {^140^^^^^133^146^if  ( data[i] == BLANK )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Variable_Misuse]^data[i] = BLANK;^141^^^^^133^146^data[i] = UNDERSCORE;^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Argument_Swapping]^for  ( datant i = 0; i < i.length; i++ )  {^139^^^^^133^146^for  ( int i = 0; i < data.length; i++ )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < data.length.length; i++ )  {^139^^^^^133^146^for  ( int i = 0; i < data.length; i++ )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Argument_Swapping]^for  ( data.lengthnt i = 0; i < i; i++ )  {^139^^^^^133^146^for  ( int i = 0; i < data.length; i++ )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= data.length; i++ )  {^139^^^^^133^146^for  ( int i = 0; i < data.length; i++ )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < data.length; i++ )  {^139^^^^^133^146^for  ( int i = 0; i < data.length; i++ )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < bytes.length; i++ )  {^139^^^^^133^146^for  ( int i = 0; i < data.length; i++ )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i == data.length; i++ )  {^139^^^^^133^146^for  ( int i = 0; i < data.length; i++ )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Variable_Misuse]^if  ( data[i] == UNDERSCORE )  {^140^^^^^133^146^if  ( data[i] == BLANK )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < data.length; i++ )  {^139^^^^^133^146^for  ( int i = 0; i < data.length; i++ )  {^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Variable_Misuse]^return bytes;^145^^^^^133^146^return data;^[CLASS] QCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  data  boolean  encodeBlanks  String  charset  byte  BLANK  UNDERSCORE  int  i  
[BugLab_Variable_Misuse]^if  ( tmp == null )  {^149^^^^^148^172^if  ( bytes == null )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Operator]^if  ( bytes != null )  {^149^^^^^148^172^if  ( bytes == null )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Literal]^boolean hasUnderscores = true;^152^^^^^148^172^boolean hasUnderscores = false;^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^if  ( tmp[i] == UNDERSCORE )  {^154^^^^^148^172^if  ( bytes[i] == UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^if  ( bytes[i] == b )  {^154^^^^^148^172^if  ( bytes[i] == UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Argument_Swapping]^if  ( UNDERSCORE[i] == bytes )  {^154^^^^^148^172^if  ( bytes[i] == UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Operator]^if  ( bytes[i] != UNDERSCORE )  {^154^^^^^148^172^if  ( bytes[i] == UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Literal]^hasUnderscores = false;^155^^^^^148^172^hasUnderscores = true;^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Argument_Swapping]^for  ( bytesnt i = 0; i < i.length; i++ )  {^153^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Argument_Swapping]^for  ( bytes.lengthnt i = 0; i < i; i++ )  {^153^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= bytes.length; i++ )  {^153^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < bytes.length; i++ )  {^153^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^if  ( encodeBlanks )  {^159^^^^^148^172^if  ( hasUnderscores )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Argument_Swapping]^if  ( UNDERSCORE != b )  {^163^^^^^148^172^if  ( b != UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Operator]^if  ( b == UNDERSCORE )  {^163^^^^^148^172^if  ( b != UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^tmp[i] = b;^166^^^^^148^172^tmp[i] = BLANK;^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^tmp[i] = UNDERSCORE;^164^^^^^148^172^tmp[i] = b;^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < tmp.length; i++ )  {^161^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < bytes.lengthytes.length; i++ )  {^161^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i > bytes.length; i++ )  {^161^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < bytes.length; i++ )  {^161^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^if  ( BLANK != UNDERSCORE )  {^163^^^^^148^172^if  ( b != UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^if  ( b != b )  {^163^^^^^148^172^if  ( b != UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Operator]^if  ( b <= UNDERSCORE )  {^163^^^^^148^172^if  ( b != UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^byte b = tmp[i];^162^^^^^148^172^byte b = bytes[i];^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^return QuotedPrintableCodec.decodeQuotedPrintable ( bytes ) ;^169^^^^^148^172^return QuotedPrintableCodec.decodeQuotedPrintable ( tmp ) ;^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^if  ( UNDERSCORE != UNDERSCORE )  {^163^^^^^148^172^if  ( b != UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^if  ( b != BLANK )  {^163^^^^^148^172^if  ( b != UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < UNDERSCOREytes.length; i++ )  {^161^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Argument_Swapping]^for  ( bytes.lengthnt i = 0; i < i; i++ )  {^161^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= bytes.length; i++ )  {^161^^^^^148^172^for  ( int i = 0; i < bytes.length; i++ )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Wrong_Operator]^if  ( b >= UNDERSCORE )  {^163^^^^^148^172^if  ( b != UNDERSCORE )  {^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^return QuotedPrintableCodec.decodeQuotedPrintable ( tmp ) ;^171^^^^^148^172^return QuotedPrintableCodec.decodeQuotedPrintable ( bytes ) ;^[CLASS] QCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] BitSet  PRINTABLE_CHARS  byte[]  bytes  tmp  boolean  encodeBlanks  hasUnderscores  String  charset  byte  BLANK  UNDERSCORE  b  int  i  
[BugLab_Variable_Misuse]^if  ( charset == null )  {^187^^^^^186^195^if  ( pString == null )  {^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] String   String pString String charset [VARIABLES] BitSet  PRINTABLE_CHARS  UnsupportedEncodingException  e  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^if  ( pString != null )  {^187^^^^^186^195^if  ( pString == null )  {^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] String   String pString String charset [VARIABLES] BitSet  PRINTABLE_CHARS  UnsupportedEncodingException  e  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Argument_Swapping]^return encodeText ( charset, pString ) ;^191^^^^^186^195^return encodeText ( pString, charset ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] String   String pString String charset [VARIABLES] BitSet  PRINTABLE_CHARS  UnsupportedEncodingException  e  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Variable_Misuse]^if  ( charset == null )  {^208^^^^^207^212^if  ( pString == null )  {^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] String   String pString [VARIABLES] BitSet  PRINTABLE_CHARS  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^if  ( pString != null )  {^208^^^^^207^212^if  ( pString == null )  {^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] String   String pString [VARIABLES] BitSet  PRINTABLE_CHARS  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Variable_Misuse]^return encode ( charset, getDefaultCharset (  )  ) ;^211^^^^^207^212^return encode ( pString, getDefaultCharset (  )  ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] String   String pString [VARIABLES] BitSet  PRINTABLE_CHARS  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Variable_Misuse]^if  ( charset == null )  {^227^^^^^226^235^if  ( pString == null )  {^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] String   String pString [VARIABLES] BitSet  PRINTABLE_CHARS  UnsupportedEncodingException  e  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^if  ( pString != null )  {^227^^^^^226^235^if  ( pString == null )  {^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] String   String pString [VARIABLES] BitSet  PRINTABLE_CHARS  UnsupportedEncodingException  e  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Variable_Misuse]^return decodeText ( charset ) ;^231^^^^^226^235^return decodeText ( pString ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] String   String pString [VARIABLES] BitSet  PRINTABLE_CHARS  UnsupportedEncodingException  e  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^if  ( pObject != null )  {^248^^^^^247^257^if  ( pObject == null )  {^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^} else if  ( pObject  >>  String )  {^250^^^^^247^257^} else if  ( pObject instanceof String )  {^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + pObject.getClass (  ||  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  ^  pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + pObject.getClass (  ==  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "   instanceof   pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + pObject.getClass (  <<  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  >  pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^} else if  ( pObject  ||  String )  {^250^^^^^247^257^} else if  ( pObject instanceof String )  {^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  !=  pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  &  pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + pObject.getClass (  <=  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  &&  pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + pObject.getClass (  >  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  <  pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^253^254^255^^^247^257^throw new EncoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be encoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] encode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^if  ( pObject != null )  {^273^^^^^272^282^if  ( pObject == null )  {^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^} else if  ( pObject  >>  String )  {^275^^^^^272^282^} else if  ( pObject instanceof String )  {^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + pObject.getClass (   instanceof   ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  >  pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + pObject.getClass (  &&  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  &  pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + pObject.getClass (  >>  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  <=  pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + pObject.getClass (  <<  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  &&  pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^} else if  ( pObject  <<  String )  {^275^^^^^272^282^} else if  ( pObject instanceof String )  {^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + pObject.getClass (  <  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + pObject.getClass (  ==  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + pObject.getClass (  >=  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^278^279^280^^^272^282^throw new DecoderException ( "Objects of type " + pObject.getClass (  ) .getName (  )  + " cannot be decoded using Q codec" ) ;^[CLASS] QCodec  [METHOD] decode [RETURN_TYPE] Object   Object pObject [VARIABLES] BitSet  PRINTABLE_CHARS  Object  pObject  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Variable_Misuse]^return pString;^290^^^^^289^291^return this.charset;^[CLASS] QCodec  [METHOD] getDefaultCharset [RETURN_TYPE] String   [VARIABLES] BitSet  PRINTABLE_CHARS  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Variable_Misuse]^return hasUnderscores;^299^^^^^298^300^return this.encodeBlanks;^[CLASS] QCodec  [METHOD] isEncodeBlanks [RETURN_TYPE] boolean   [VARIABLES] BitSet  PRINTABLE_CHARS  boolean  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
[BugLab_Variable_Misuse]^this.encodeBlanks = hasUnderscores;^309^^^^^308^310^this.encodeBlanks = b;^[CLASS] QCodec  [METHOD] setEncodeBlanks [RETURN_TYPE] void   boolean b [VARIABLES] BitSet  PRINTABLE_CHARS  boolean  b  encodeBlanks  hasUnderscores  String  charset  pString  byte  BLANK  UNDERSCORE  b  
