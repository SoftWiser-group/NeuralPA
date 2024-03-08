[BugLab_Variable_Misuse]^private String charset = 1;^51^^^^^46^56^private String charset = CharacterEncodingNames.UTF8;^[CLASS] BCodec   [VARIABLES] 
[BugLab_Wrong_Operator]^if  ( bytes != null )  {^79^^^^^78^83^if  ( bytes == null )  {^[CLASS] BCodec  [METHOD] doEncoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] byte[]  bytes  String  charset  boolean  
[BugLab_Wrong_Operator]^if  ( bytes != null )  {^86^^^^^85^90^if  ( bytes == null )  {^[CLASS] BCodec  [METHOD] doDecoding [RETURN_TYPE] byte[]   byte[] bytes [VARIABLES] byte[]  bytes  String  charset  boolean  
[BugLab_Variable_Misuse]^if  ( charset == null )  {^105^^^^^104^113^if  ( value == null )  {^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] String   String value String charset [VARIABLES] UnsupportedEncodingException  e  String  charset  value  boolean  
[BugLab_Wrong_Operator]^if  ( value != null )  {^105^^^^^104^113^if  ( value == null )  {^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] String   String value String charset [VARIABLES] UnsupportedEncodingException  e  String  charset  value  boolean  
[BugLab_Argument_Swapping]^return encodeText ( charset, value ) ;^109^^^^^104^113^return encodeText ( value, charset ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] String   String value String charset [VARIABLES] UnsupportedEncodingException  e  String  charset  value  boolean  
[BugLab_Variable_Misuse]^if  ( charset == null )  {^126^^^^^125^130^if  ( value == null )  {^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] String   String value [VARIABLES] String  charset  value  boolean  
[BugLab_Wrong_Operator]^if  ( value != null )  {^126^^^^^125^130^if  ( value == null )  {^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] String   String value [VARIABLES] String  charset  value  boolean  
[BugLab_Variable_Misuse]^return encode ( charset, getDefaultCharset (  )  ) ;^129^^^^^125^130^return encode ( value, getDefaultCharset (  )  ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] String   String value [VARIABLES] String  charset  value  boolean  
[BugLab_Variable_Misuse]^if  ( charset == null )  {^145^^^^^144^153^if  ( value == null )  {^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] String   String value [VARIABLES] UnsupportedEncodingException  e  String  charset  value  boolean  
[BugLab_Wrong_Operator]^if  ( value != null )  {^145^^^^^144^153^if  ( value == null )  {^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] String   String value [VARIABLES] UnsupportedEncodingException  e  String  charset  value  boolean  
[BugLab_Variable_Misuse]^return decodeText ( charset ) ;^149^^^^^144^153^return decodeText ( value ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] String   String value [VARIABLES] UnsupportedEncodingException  e  String  charset  value  boolean  
[BugLab_Wrong_Operator]^if  ( value != null )  {^166^^^^^165^175^if  ( value == null )  {^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^} else if  ( value  <<  String )  {^168^^^^^165^175^} else if  ( value instanceof String )  {^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + value.getClass (  &&  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "   instanceof   value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + value.getClass (  !=  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  ||  value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + value.getClass (   instanceof   ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + value.getClass (  |  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^} else if  ( value  &  String )  {^168^^^^^165^175^} else if  ( value instanceof String )  {^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + value.getClass (  ==  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  &  value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + value.getClass (  &  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type " + value.getClass (  <  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new EncoderException ( "Objects of type "  >  value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^171^172^173^^^165^175^throw new EncoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be encoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] encode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^if  ( value != null )  {^191^^^^^190^200^if  ( value == null )  {^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^} else if  ( value  <  String )  {^193^^^^^190^200^} else if  ( value instanceof String )  {^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + value.getClass (  |  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  <=  value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + value.getClass (  >=  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  ==  value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  <<  value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + value.getClass (  !=  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  >  value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^} else if  ( value  ||  String )  {^193^^^^^190^200^} else if  ( value instanceof String )  {^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + value.getClass (  ==  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  &  value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + value.getClass (  ^  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  <  value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type " + value.getClass (  <=  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Wrong_Operator]^throw new DecoderException ( "Objects of type "  !=  value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^196^197^198^^^190^200^throw new DecoderException ( "Objects of type " + value.getClass (  ) .getName (  )  + " cannot be decoded using BCodec" ) ;^[CLASS] BCodec  [METHOD] decode [RETURN_TYPE] Object   Object value [VARIABLES] Object  value  String  charset  value  boolean  
[BugLab_Variable_Misuse]^return value;^208^^^^^207^209^return this.charset;^[CLASS] BCodec  [METHOD] getDefaultCharset [RETURN_TYPE] String   [VARIABLES] String  charset  value  boolean  