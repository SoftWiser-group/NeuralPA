[BugLab_Argument_Swapping]^if  ( in == null || archiverName == null )  {^59^^^^^56^75^if  ( archiverName == null || in == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   String archiverName InputStream in [VARIABLES] boolean  InputStream  in  String  archiverName  
[BugLab_Wrong_Operator]^if  ( archiverName == null && in == null )  {^59^^^^^56^75^if  ( archiverName == null || in == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   String archiverName InputStream in [VARIABLES] boolean  InputStream  in  String  archiverName  
[BugLab_Wrong_Operator]^if  ( archiverName != null || in == null )  {^59^^^^^56^75^if  ( archiverName == null || in == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   String archiverName InputStream in [VARIABLES] boolean  InputStream  in  String  archiverName  
[BugLab_Wrong_Operator]^if  ( archiverName == null || in != null )  {^59^^^^^56^75^if  ( archiverName == null || in == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   String archiverName InputStream in [VARIABLES] boolean  InputStream  in  String  archiverName  
[BugLab_Argument_Swapping]^if  ( out == null || archiverName == null )  {^89^^^^^86^106^if  ( archiverName == null || out == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveOutputStream [RETURN_TYPE] ArchiveOutputStream   String archiverName OutputStream out [VARIABLES] boolean  OutputStream  out  String  archiverName  
[BugLab_Wrong_Operator]^if  ( archiverName == null && out == null )  {^89^^^^^86^106^if  ( archiverName == null || out == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveOutputStream [RETURN_TYPE] ArchiveOutputStream   String archiverName OutputStream out [VARIABLES] boolean  OutputStream  out  String  archiverName  
[BugLab_Wrong_Operator]^if  ( archiverName != null || out == null )  {^89^^^^^86^106^if  ( archiverName == null || out == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveOutputStream [RETURN_TYPE] ArchiveOutputStream   String archiverName OutputStream out [VARIABLES] boolean  OutputStream  out  String  archiverName  
[BugLab_Wrong_Operator]^if  ( archiverName == null || out != null )  {^89^^^^^86^106^if  ( archiverName == null || out == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveOutputStream [RETURN_TYPE] ArchiveOutputStream   String archiverName OutputStream out [VARIABLES] boolean  OutputStream  out  String  archiverName  
[BugLab_Wrong_Operator]^if  ( in != null )  {^119^^^^^104^134^if  ( in == null )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Wrong_Literal]^final byte[] signature = new byte[signatureLength];^127^^^^^112^142^final byte[] signature = new byte[12];^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Variable_Misuse]^in.mark ( signatureLength ) ;^128^^^^^113^143^in.mark ( signature.length ) ;^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Argument_Swapping]^in.mark ( signature ) ;^128^^^^^113^143^in.mark ( signature.length ) ;^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Argument_Swapping]^if  ( ZipArchiveInputStream.matches ( signatureLength, signature )  )  {^132^^^^^117^147^if  ( ZipArchiveInputStream.matches ( signature, signatureLength )  )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Argument_Swapping]^} else if  ( ArArchiveInputStream.matches ( signatureLength, signature )  )  {^140^^^^^125^155^} else if  ( ArArchiveInputStream.matches ( signature, signatureLength )  )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Argument_Swapping]^} else if  ( CpioArchiveInputStream.matches ( signatureLength, signature )  )  {^142^143^^^^127^157^} else if  ( CpioArchiveInputStream.matches ( signature, signatureLength )  )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Argument_Swapping]^} else if  ( TarArchiveInputStream .matches ( signatureLength, signature )  )  {^137^138^^^^122^152^} else if  ( TarArchiveInputStream .matches ( signature, signatureLength )  )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Argument_Swapping]^} else if  ( JarArchiveInputStream .matches ( signatureLength, signature )  )  {^134^135^^^^119^149^} else if  ( JarArchiveInputStream .matches ( signature, signatureLength )  )  {^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
[BugLab_Argument_Swapping]^int inLength = signature.read ( signature ) ;^130^^^^^115^145^int signatureLength = in.read ( signature ) ;^[CLASS] ArchiveStreamFactory  [METHOD] createArchiveInputStream [RETURN_TYPE] ArchiveInputStream   InputStream in [VARIABLES] byte[]  signature  boolean  InputStream  in  IOException  e  int  signatureLength  
